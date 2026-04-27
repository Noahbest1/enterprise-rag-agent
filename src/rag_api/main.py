"""Minimal FastAPI surface for the new RAG pipeline.

Endpoints:
    GET  /health
    GET  /kbs                 -- list knowledge bases
    POST /kbs                 -- create empty KB
    POST /kbs/{kb_id}/build   -- reindex a KB from data/kb/<kb_id>/raw/
    POST /search              -- retrieval only
    POST /answer              -- end-to-end RAG answer

Kept tiny on purpose; pairs 1:1 with pipeline.py.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import json

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from rag.logging import bind_trace_id, configure_logging, get_logger
from rag.observability import get_tracer, init_tracing, metrics_payload, record_request

from .audit import record_http_audit_from_middleware
from .auth import AuthContext, require_api_key
from .invoice_pdf import render_invoice_pdf
from .rate_limit import rate_limit_middleware

configure_logging()
init_tracing(service_name="rag-api")
_tracer = get_tracer("rag_api")
log = get_logger("rag_api")

_UI_PATH = Path(__file__).resolve().parent / "ui.html"

from rag import knowledge_base as kb_mod
from rag.config import settings
from rag.index.build import build_indexes
from rag.ingest.pipeline import ingest_directory, write_chunks_jsonl
from rag.pipeline import answer_query, answer_query_stream
from rag.retrieval.hybrid import retrieve
from rag.query.normalize import normalize_query
from rag.query.rewrite import rewrite_query

from .admin_routes import router as admin_router
from .agent_routes import router as agent_router
from .compat import router as compat_router
from .user_events import router as user_events_router
from .vision_routes import router as vision_router


app = FastAPI(title="RAG API", version="2.0.0")

origins = settings.cors_allow_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.middleware("http")(rate_limit_middleware)


@app.middleware("http")
async def trace_and_log(request: Request, call_next):
    """Attach trace_id to every request; log inbound + outbound with timing."""
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
    bind_trace_id(trace_id)
    t0 = time.perf_counter()
    log.info(
        "http_request_start",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
    )
    status_code = 500
    # Wrap the whole request in an OTel root span. When OTEL isn't configured
    # this is a no-op tracer so the overhead is a function call.
    with _tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.route", request.url.path)
        span.set_attribute("rag.trace_id", trace_id)
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            span.set_attribute("http.status_code", status_code)
        except Exception as e:
            span.record_exception(e)
            log.exception("http_request_error", error=str(e))
            raise
        finally:
            route = request.scope.get("route")
            path_template = getattr(route, "path", None) or request.url.path
            latency_s = time.perf_counter() - t0
            record_request(
                method=request.method,
                path=path_template,
                status=status_code,
                latency_s=latency_s,
            )
    response.headers["x-trace-id"] = trace_id
    latency_ms_int = int(latency_s * 1000)
    log.info(
        "http_request_end",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms_int,
    )
    # Extension-stage: persist an audit row for every non-exempt request.
    # Runs after the response is composed so status + latency are accurate.
    # Failures are swallowed inside record_audit so audit can't take the API down.
    try:
        record_http_audit_from_middleware(
            request=request,
            status_code=response.status_code,
            latency_ms=latency_ms_int,
            trace_id=trace_id,
        )
    except Exception as e:  # pragma: no cover
        log.warning("audit_hook_failed", error=str(e))
    return response


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint. Text exposition format."""
    body, content_type = metrics_payload()
    return Response(content=body, media_type=content_type)


class SearchRequest(BaseModel):
    query: str
    kb_id: str
    top_k: int | None = None
    use_rewrite: bool = True
    use_rerank: bool = True


class FeedbackRequest(BaseModel):
    trace_id: str
    kb_id: str
    query: str
    answer: str
    verdict: str  # "up" | "down"
    reason: str | None = None
    user_id: str | None = None


@app.post("/feedback")
def post_feedback(req: FeedbackRequest, auth: AuthContext = Depends(require_api_key)):
    """Persist a thumbs up/down verdict. Feeds the Day 9 eval regression pipeline."""
    from rag.db.base import SessionLocal
    from rag.db.models import Feedback

    if req.verdict not in ("up", "down"):
        raise HTTPException(status_code=400, detail="verdict must be 'up' or 'down'")

    with SessionLocal() as s:
        row = Feedback(
            trace_id=req.trace_id,
            kb_id=req.kb_id,
            query=req.query,
            answer=req.answer,
            verdict=req.verdict,
            reason=req.reason,
            user_id=req.user_id,
        )
        s.add(row)
        s.commit()
        log.info(
            "feedback_saved",
            id=row.id,
            kb_id=req.kb_id,
            verdict=req.verdict,
            trace_id_feedback=req.trace_id,
        )
        # Business-event audit: lets ops filter /audit?event_type=feedback_received
        from .audit import hash16, record_audit
        record_audit(
            event_type="feedback_received",
            trace_id=req.trace_id,
            tenant_id=auth.tenant_id,
            api_key_id=auth.api_key_id,
            user_id=req.user_id,
            query_hash=hash16(req.query),
            answer_hash=hash16(req.answer),
            extra={"kb_id": req.kb_id, "verdict": req.verdict, "feedback_id": row.id},
        )
        return {"ok": True, "id": row.id}


class AnswerRequest(BaseModel):
    query: str
    kb_id: str
    use_rewrite: bool = True
    use_rerank: bool = True
    use_multi_query: bool = False
    # Optional multi-turn history -- ordered oldest to newest.
    # Each item: {"role": "user" | "assistant", "content": str}.
    conversation: list[dict] | None = None
    # Optional metadata filter: source_path_contains / title_contains /
    # section_contains / source_type.
    filter: dict | None = None


# The compat router owns POST /answer (it handles both old and new request
# shapes). We register it before the native /answer handler below so the
# legacy endpoint wins. The native GET /kbs etc. are untouched.
app.include_router(compat_router)
app.include_router(agent_router)
app.include_router(vision_router)
app.include_router(admin_router)
app.include_router(user_events_router)


class CreateKBRequest(BaseModel):
    kb_id: str
    description: str = ""


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    # Shape matches the old frontend's HealthResponse so the status card lights up.
    import uuid
    from datetime import datetime, timezone

    return {
        "status": "ok",
        "service": "RAG API",
        "version": "2.0.0",
        "qwen_configured": bool(settings.qwen_api_key),
        "default_answer_mode": "extractive_grounded_v1",
        "fallback_answer_mode": "extractive_grounded_v1",
        "answer_fallback_enabled": True,
        "log_path": "",
        "kb_root": str(settings.kb_root),
        "trace_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/kbs")
def list_kbs():
    return {"items": [kb.to_dict() for kb in kb_mod.list_kbs()]}


@app.post("/kbs")
def create_kb(req: CreateKBRequest):
    kb = kb_mod.create_kb(req.kb_id, description=req.description)
    return kb.to_dict()


@app.post("/kbs/{kb_id}/build")
def build_kb(kb_id: str):
    try:
        kb = kb_mod.get_kb(kb_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    raw_dir = kb.root / "raw"
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise HTTPException(status_code=400, detail=f"No files in {raw_dir}")
    chunks = ingest_directory(raw_dir, kb_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from raw files")
    write_chunks_jsonl(chunks, kb.chunks_path)
    result = build_indexes(kb.root, kb_id)
    return {"kb_id": kb_id, "chunk_count": result.chunk_count}


# Resume-grade upload-to-KB endpoint: one shot from "I have a PDF" to
# "I can ask questions about it". Frontend uploads multipart files, we
# write them to data/kb/<kb_id>/raw/, then run the same build pipeline.
# Files with unsupported extensions are skipped (not 400'd) so a mixed
# bundle still indexes the supported ones; we report both lists in the
# response so the UI can show the user what landed and what was skipped.
@app.post("/kbs/{kb_id}/upload")
async def upload_to_kb(
    kb_id: str,
    files: list[UploadFile] = File(...),
    create_if_missing: bool = Form(default=True),
    description: str = Form(default=""),
):
    """Upload N files into <kb_id>/raw/ and (re)build the indexes.

    - If the KB doesn't exist and ``create_if_missing=true``, create it first.
    - Files with extensions not in ``loaders.supported_extensions()`` are
      saved but won't be ingested (we skip rather than fail so a mixed
      drop-zone still works).
    - 8 MB cap per file to match the vision endpoints; the API gateway
      should set its own request-size limit in production.
    """
    from rag.ingest.loaders import supported_extensions
    MAX_BYTES = 8 * 1024 * 1024

    try:
        kb = kb_mod.get_kb(kb_id)
    except FileNotFoundError:
        if not create_if_missing:
            raise HTTPException(status_code=404, detail=f"KB '{kb_id}' not found")
        kb = kb_mod.create_kb(kb_id, description=description)

    raw_dir = kb.root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    exts = supported_extensions()

    saved: list[dict] = []
    skipped: list[dict] = []
    for f in files:
        name = (f.filename or "").strip() or "untitled"
        # Sanitize: drop any path traversal, keep just the basename.
        safe_name = Path(name).name
        if not safe_name:
            skipped.append({"filename": name, "reason": "empty filename"})
            continue
        suffix = Path(safe_name).suffix.lower()
        if suffix not in exts:
            skipped.append({"filename": safe_name, "reason": f"unsupported extension {suffix!r}"})
            continue
        body = await f.read()
        if len(body) > MAX_BYTES:
            skipped.append({"filename": safe_name, "reason": f"too large ({len(body)} > {MAX_BYTES})"})
            continue
        if not body:
            skipped.append({"filename": safe_name, "reason": "empty file"})
            continue
        dest = raw_dir / safe_name
        dest.write_bytes(body)
        saved.append({"filename": safe_name, "bytes": len(body)})

    if not saved:
        raise HTTPException(
            status_code=400,
            detail={"message": "no files saved", "skipped": skipped},
        )

    chunks = ingest_directory(raw_dir, kb_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks produced from saved files")
    write_chunks_jsonl(chunks, kb.chunks_path)
    result = build_indexes(kb.root, kb_id)
    return {
        "kb_id": kb_id,
        "saved": saved,
        "skipped": skipped,
        "chunk_count": result.chunk_count,
        "indexed_chunks": result.indexed_chunks,
        "vector_backend": result.vector_backend,
    }


class StreamAnswerRequest(BaseModel):
    query: str
    kb_id: str
    use_rewrite: bool = True
    use_rerank: bool = True
    conversation: list[dict] | None = None
    filter: dict | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/answer/stream")
async def answer_stream(
    req: StreamAnswerRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """SSE-stream the LLM answer. Events: meta / hits / delta / abstain / done / error."""
    async def _gen():
        try:
            async for event_type, data in answer_query_stream(
                req.query, req.kb_id,
                use_rewrite=req.use_rewrite,
                use_rerank=req.use_rerank,
                conversation=req.conversation,
                filter=req.filter,
            ):
                yield _sse(event_type, data)
        except FileNotFoundError as e:
            yield _sse("error", {"detail": str(e)})
        except Exception as e:
            log.exception("answer_stream_failed", error=str(e))
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # hint to nginx/cloudfront: don't buffer
        },
    )


# ---------- Tier-2 interactive-product endpoints ----------

@app.get("/invoice/{invoice_id}.pdf")
def download_invoice_pdf(invoice_id: int):
    """Serve a real PDF rendered from the invoice row.

    No auth gate on this route so the link rendered in chat works when the
    user clicks it from the bubble (they've already been authed at chat time).
    Production would sign the URL with an HMAC + TTL.
    """
    from rag.db.base import SessionLocal
    from rag.db.models import Invoice
    from agent.tools.invoices import _invoice_to_dict
    from agent.tools.orders import get_order

    with SessionLocal() as s:
        inv = s.get(Invoice, invoice_id)
        if inv is None:
            raise HTTPException(status_code=404, detail="invoice not found")
        inv_dict = _invoice_to_dict(inv)

    order = get_order(inv_dict["order_id"]) if inv_dict.get("order_id") else None
    pdf_bytes = render_invoice_pdf(inv_dict, order)
    filename = f"invoice-{invoice_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


class ConfirmReturnRequest(BaseModel):
    order_id: str
    reason: str = "用户确认退货"


@app.post("/agent/actions/confirm-return")
def action_confirm_return(
    req: ConfirmReturnRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """User clicks the '确认提交退货申请' button in the agent chat."""
    from agent.tools.returns import check_eligibility, create_return_request
    elig = check_eligibility(req.order_id)
    if not elig.get("ok"):
        raise HTTPException(status_code=400, detail=elig.get("reason") or "not eligible")
    rr = create_return_request(req.order_id, kind="return", reason=req.reason)
    from .audit import record_audit
    record_audit(
        event_type="return_request_created",
        tenant_id=elig.get("tenant"), user_id=None,
        extra={
            "order_id": req.order_id,
            "refund_cents": elig.get("refund_cents"),
            "request_id": rr.get("id"),
        },
    )
    return {"ok": True, "request": rr, "eligibility": elig}


class CancelOrderRequest(BaseModel):
    order_id: str


@app.post("/agent/actions/cancel-order")
def action_cancel_order(
    req: CancelOrderRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """User clicks '取消订单' on a not-yet-shipped order. Self-service path
    that bypasses the return / complaint flow because the merchandise has
    not left the warehouse — no logistics or human intervention needed.

    Returns 409 if the order is past the cancellable window (shipped /
    delivered / refunded / already cancelled). The UI is expected to
    fall back to the complaint-escalation flow in those cases.
    """
    from agent.tools.orders import cancel_order
    result = cancel_order(req.order_id)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("reason") or "not_cancellable")

    from .audit import record_audit
    record_audit(
        event_type="order_cancelled",
        tenant_id=None, user_id=None,
        extra={
            "order_id": req.order_id,
            "previous_status": result.get("previous_status"),
            "refunded_cents": result.get("refunded_cents"),
        },
    )
    return {"ok": True, "result": result}


class SetDefaultAddressRequest(BaseModel):
    user_id: str
    address_id: int


@app.post("/agent/actions/set-default-address")
def action_set_default_address(
    req: SetDefaultAddressRequest,
    auth: AuthContext = Depends(require_api_key),
):
    from agent.tools.accounts import set_default_address
    updated = set_default_address(req.user_id, req.address_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="address not found or not owned by user")
    return {"ok": True, "address": updated}


class ReopenReturnRequest(BaseModel):
    request_id: int


@app.post("/agent/actions/reopen-return-request")
def action_reopen_return_request(
    req: ReopenReturnRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """User changes their mind on a cancelled return request.

    Flips status back to ``pending`` so the request resumes processing.
    Mirrors ``/agent/actions/reopen-complaint``: state-machine entities
    on this project all support a reversible "user closed by mistake"
    path so the user never gets stuck with a dead row.
    """
    from agent.tools.returns import reopen_return_request
    try:
        updated = reopen_return_request(req.request_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="return request not found")
    from .audit import record_audit
    record_audit(
        event_type="return_request_reopened",
        tenant_id=auth.tenant_id, user_id=None,
        extra={"request_id": req.request_id, "new_status": updated["status"]},
    )
    return {"ok": True, "request": updated}


class CancelReturnRequest(BaseModel):
    request_id: int


@app.post("/agent/actions/cancel-return-request")
def action_cancel_return_request(
    req: CancelReturnRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """User clicks '取消此退货' on an existing ReturnRequest card."""
    from rag.db.base import SessionLocal
    from rag.db.models import ReturnRequest
    with SessionLocal() as s:
        rr = s.get(ReturnRequest, req.request_id)
        if rr is None:
            raise HTTPException(status_code=404, detail="return_request not found")
        if rr.status in ("cancelled", "completed", "rejected"):
            raise HTTPException(status_code=400,
                                detail=f"not cancellable (current status: {rr.status})")
        rr.status = "cancelled"
        s.commit()
    from .audit import record_audit
    record_audit(
        event_type="return_request_cancelled",
        tenant_id=auth.tenant_id, user_id=None,
        extra={"request_id": req.request_id},
    )
    return {"ok": True, "request_id": req.request_id, "status": "cancelled"}


class SubmitComplaintRequest(BaseModel):
    """Body for the user clicking "提交工单" on the complaint preview card.

    Mirrors the dry-run preview shape the specialist emits. The ``content``
    field is the user's raw utterance — never persisted, only sha256[:16]'d
    for the audit row inside ``create_complaint``.
    """
    severity: str  # "high" | "medium" | "low"
    topic: str     # "delivery" | "quality" | "service" | "refund" | "price" | "other"
    user_id: str
    tenant: str
    order_id: str | None = None
    content: str
    # ChatSession that filed this complaint. Optional — legacy callers
    # without sessions still work; the row's thread_id stays NULL.
    thread_id: str | None = None


@app.post("/agent/actions/submit-complaint")
def action_submit_complaint(
    req: SubmitComplaintRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """Sole DB-write path for complaints.

    The complaint specialist only emits a preview; this endpoint is the
    one place where a row actually lands in ``complaints``. High-severity
    submissions also write a ``complaint_escalated`` audit row (content
    hashed only — same policy as everywhere).
    """
    if req.severity not in ("high", "medium", "low"):
        raise HTTPException(status_code=400, detail=f"bad severity: {req.severity!r}")
    valid_topics = {"delivery", "quality", "service", "refund", "price", "other"}
    if req.topic not in valid_topics:
        raise HTTPException(status_code=400, detail=f"bad topic: {req.topic!r}")

    from agent.tools.complaints import create_complaint
    complaint = create_complaint(
        user_id=req.user_id,
        tenant=req.tenant,
        order_id=req.order_id,
        topic=req.topic,
        severity=req.severity,
        content=req.content,
    )
    # Link the complaint to its originating ChatSession so the user's
    # session sidebar can show "tickets created here". Done as a follow-up
    # write (not a create_complaint param) to keep the tool function's
    # signature unchanged for non-API callers.
    if req.thread_id and complaint.get("id"):
        from rag.db.base import SessionLocal as DbSession
        from rag.db.models import Complaint as ComplaintModel
        with DbSession() as s:
            row = s.get(ComplaintModel, complaint["id"])
            if row is not None:
                row.thread_id = req.thread_id
                s.commit()
        complaint["thread_id"] = req.thread_id

    # Audit breadcrumb for high-severity tickets so ops can alert on them.
    if complaint.get("escalated"):
        from .audit import hash16, record_audit
        record_audit(
            event_type="complaint_escalated",
            tenant_id=req.tenant,
            user_id=req.user_id,
            extra={
                "complaint_id": complaint["id"],
                "topic": req.topic,
                "severity": req.severity,
                "assigned_to": complaint.get("assigned_to"),
                "order_id": req.order_id,
                "content_hash": hash16(req.content),
                "via": "user_button",  # vs old auto-write path
            },
        )
    return {"ok": True, "complaint": complaint}


class ReopenComplaintRequest(BaseModel):
    complaint_id: int


@app.post("/agent/actions/reopen-complaint")
def action_reopen_complaint(
    req: ReopenComplaintRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """User changes their mind on a closed complaint.

    Restores the appropriate active state based on the original severity
    (high → escalated + new assignee + fresh 1h SLA; otherwise → open with
    standard SLA window). Audit row records the reopen so ops can spot
    "ping-pong" tickets that get closed and reopened repeatedly.
    """
    from agent.tools.complaints import reopen_complaint
    try:
        updated = reopen_complaint(req.complaint_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="complaint not found")
    from .audit import record_audit
    record_audit(
        event_type="complaint_reopened",
        tenant_id=updated.get("tenant"), user_id=None,
        extra={
            "complaint_id": req.complaint_id,
            "new_status": updated["status"],
            "new_assigned_to": updated.get("assigned_to"),
            "severity": updated.get("severity"),
        },
    )
    return {"ok": True, "complaint": updated}


class CancelComplaintRequest(BaseModel):
    complaint_id: int


@app.post("/agent/actions/cancel-complaint")
def action_cancel_complaint(
    req: CancelComplaintRequest,
    auth: AuthContext = Depends(require_api_key),
):
    from rag.db.base import SessionLocal
    from rag.db.models import Complaint
    with SessionLocal() as s:
        c = s.get(Complaint, req.complaint_id)
        if c is None:
            raise HTTPException(status_code=404, detail="complaint not found")
        c.status = "closed"
        # Also clear the escalation flag — the user pressing "撤销升级并关闭"
        # explicitly de-escalates, so the admin's "only escalated" filter
        # should drop the row from their queue. Without this, closed-but-
        # escalated rows lingered in admin's view and admins kept replying.
        c.escalated = False
        s.commit()
    from .audit import record_audit
    record_audit(
        event_type="complaint_cancelled",
        tenant_id=auth.tenant_id, user_id=None,
        extra={"complaint_id": req.complaint_id},
    )
    return {"ok": True, "complaint_id": req.complaint_id, "status": "closed", "escalated": False}


class VerifyPhoneChangeRequest(BaseModel):
    user_id: str
    new_phone: str
    code: str


# Demo-only fixed code so the flow can be exercised without a real SMS gateway.
_DEMO_VERIFICATION_CODE = "123456"


@app.post("/agent/actions/verify-phone-change")
def action_verify_phone_change(
    req: VerifyPhoneChangeRequest,
    auth: AuthContext = Depends(require_api_key),
):
    if req.code != _DEMO_VERIFICATION_CODE:
        raise HTTPException(status_code=400, detail="invalid verification code")
    from rag.db.base import SessionLocal
    from rag.db.models import User
    with SessionLocal() as s:
        u = s.get(User, req.user_id)
        if u is None:
            raise HTTPException(status_code=404, detail="user not found")
        old_phone = u.phone
        u.phone = req.new_phone
        s.commit()
    from .audit import hash16, record_audit
    record_audit(
        event_type="phone_changed",
        tenant_id=auth.tenant_id, user_id=req.user_id,
        extra={"old_phone_hash": hash16(old_phone), "new_phone_hash": hash16(req.new_phone)},
    )
    return {"ok": True, "user_id": req.user_id, "phone_masked": _mask_phone(req.new_phone)}


def _mask_phone(phone: str) -> str:
    if not phone or len(phone) < 7:
        return "****"
    return phone[:3] + "*" * (len(phone) - 7) + phone[-4:]


class UserReplyRequest(BaseModel):
    user_id: str
    content: str


@app.post("/complaints/{complaint_id}/user-reply")
def user_reply_to_complaint(complaint_id: int, req: UserReplyRequest):
    """User-side reply on a complaint thread.

    Mirrors the admin reply flow but with ``author_kind="user"`` so the
    admin dashboard can render the message on the correct side. Validates
    that the user actually owns this complaint (rejecting cross-user
    spoofing in this single-tenant demo).
    """
    from agent.tools.complaints import add_reply, get_complaint
    from .audit import hash16, record_audit

    parent = get_complaint(complaint_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="complaint not found")
    if parent.get("user_id") and parent["user_id"] != req.user_id:
        # Don't let user X write into user Y's ticket thread.
        raise HTTPException(status_code=404, detail="complaint not found")

    reply = add_reply(
        complaint_id,
        content=req.content,
        author_label=req.user_id,
        author_kind="user",
    )
    if reply is None:
        raise HTTPException(status_code=400, detail="empty reply or complaint gone")

    record_audit(
        event_type="complaint_user_replied",
        tenant_id=parent.get("tenant"),
        user_id=req.user_id,
        extra={
            "complaint_id": complaint_id,
            "reply_id": reply["id"],
            "content_hash": hash16(req.content),
        },
    )
    return {"ok": True, "reply": reply}


class SetPreferenceRequest(BaseModel):
    key: str
    value: str
    source: str = "user"


@app.get("/users/{user_id}/preferences")
def list_user_preferences(user_id: str):
    """4th-layer agent memory: list all preferences for a user."""
    from agent.tools.preferences import list_preferences
    items = list_preferences(user_id)
    return {"user_id": user_id, "items": items}


@app.put("/users/{user_id}/preferences")
def set_user_preference(user_id: str, req: SetPreferenceRequest):
    """Upsert one preference. Source defaults to ``user`` (user explicitly
    set it via natural-language); ``inferred`` and ``admin`` are alternatives."""
    from agent.tools.preferences import set_preference
    if req.source not in ("user", "inferred", "admin"):
        raise HTTPException(status_code=400, detail=f"bad source: {req.source!r}")
    try:
        saved = set_preference(user_id, req.key, req.value, source=req.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "preference": saved}


@app.delete("/users/{user_id}/preferences/{key}")
def delete_user_preference(user_id: str, key: str):
    from agent.tools.preferences import delete_preference
    if not delete_preference(user_id, key):
        raise HTTPException(status_code=404, detail="preference not found")
    return {"ok": True}


@app.get("/users/{user_id}/sessions")
def list_user_sessions(user_id: str, limit: int = 50):
    """List a user's past chat sessions, newest first.

    Powers the ChatGPT-style session sidebar in the frontend. Each entry
    has the metadata you'd render in the sidebar: thread_id, title,
    last_msg_at, plus tenant. The actual messages live in the LangGraph
    checkpoint (``data/langgraph.sqlite``); fetch them by sending the
    thread_id to ``/agent/chat`` and the AsyncSqliteSaver restores state.
    """
    from agent.tools.sessions import list_sessions_for_user
    items = list_sessions_for_user(user_id, limit=limit)
    return {"user_id": user_id, "items": items}


class RenameSessionRequest(BaseModel):
    title: str


@app.patch("/sessions/{thread_id}")
def rename_session_endpoint(thread_id: str, req: RenameSessionRequest):
    from agent.tools.sessions import rename_session
    updated = rename_session(thread_id, title=req.title)
    if updated is None:
        raise HTTPException(status_code=404, detail="session not found or empty title")
    return {"ok": True, "session": updated}


@app.delete("/sessions/{thread_id}")
def delete_session_endpoint(thread_id: str):
    """Delete a session: removes the metadata row, clears the LangGraph
    checkpoint for this thread_id, and NULLs ``complaints.thread_id`` so
    those complaints stay visible to the user but are no longer linked
    to the deleted session. Tickets themselves are NOT deleted (they're
    user property, not session property).
    """
    from agent.tools.sessions import delete_session, get_session
    from rag.db.base import SessionLocal as DbSession
    from rag.db.models import Complaint

    if get_session(thread_id) is None:
        raise HTTPException(status_code=404, detail="session not found")

    # 1. NULL out complaints.thread_id (keep the complaints, just unlink).
    with DbSession() as s:
        s.query(Complaint).filter(Complaint.thread_id == thread_id).update(
            {"thread_id": None}, synchronize_session=False
        )
        s.commit()

    # 2. Clear the LangGraph checkpoint for this thread.
    # AsyncSqliteSaver uses a `checkpoints` table keyed on thread_id, plus
    # auxiliary tables (writes, blobs). Delete from all of them.
    try:
        import sqlite3
        from agent.graph import _DEFAULT_CHECKPOINT_PATH
        if _DEFAULT_CHECKPOINT_PATH.exists():
            conn = sqlite3.connect(str(_DEFAULT_CHECKPOINT_PATH))
            try:
                # Tables vary slightly across LangGraph versions; delete from
                # any that exist.
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = {row[0] for row in cur.fetchall()}
                for t in ("checkpoints", "writes", "blobs"):
                    if t in tables:
                        conn.execute(f"DELETE FROM {t} WHERE thread_id = ?", (thread_id,))
                conn.commit()
            finally:
                conn.close()
    except Exception:  # pragma: no cover
        pass

    # 3. Delete the session metadata row.
    delete_session(thread_id)

    return {"ok": True, "thread_id": thread_id}


@app.get("/users/{user_id}/inbox")
def user_inbox(
    user_id: str,
    since: str | None = None,
    limit: int = 50,
    thread_id: str | None = None,
):
    """User-side rehydration endpoint.

    AgentChat calls this on mount to recover state that an in-memory
    SSE subscription can't restore: complaints this user filed earlier
    + admin replies that landed while no live subscriber was watching.
    Returns newest replies first, capped at ``limit``. ``since`` (ISO-8601)
    filters to replies created after that timestamp so subsequent calls
    only return the new tail.

    Shape::

        {
          "user_id": "...",
          "items": [
            {
              "complaint_id": 12,
              "severity": "high",
              "topic": "delivery",
              "status": "escalated",
              "complaint_created_at": "...",
              "replies": [
                {"id": 3, "author_label": "jd-cs-senior-A",
                 "content": "...", "created_at": "..."},
                ...
              ]
            },
            ...
          ]
        }
    """
    from datetime import datetime, timezone
    from sqlalchemy import select
    from rag.db.base import SessionLocal
    from rag.db.models import Complaint, ComplaintReply

    def _iso_utc(dt: datetime | None) -> str | None:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"bad 'since': {since!r}")

    with SessionLocal() as s:
        # All complaints owned by this user, newest first. When `thread_id`
        # is supplied we narrow STRICTLY to that session — the panel header
        # says "本会话工单", so leaking NULL-thread_id legacy rows into every
        # session contradicted the label and confused users. Legacy complaints
        # without a session are still visible via admin / unfiltered inbox.
        stmt = select(Complaint).where(Complaint.user_id == user_id)
        if thread_id is not None:
            stmt = stmt.where(Complaint.thread_id == thread_id)
        complaints = s.execute(
            stmt.order_by(Complaint.created_at.desc())
            .limit(max(1, min(int(limit), 200)))
        ).scalars().all()
        complaint_ids = [c.id for c in complaints]
        replies_by_complaint: dict[int, list] = {}
        if complaint_ids:
            rep_rows = s.execute(
                select(ComplaintReply)
                .where(ComplaintReply.complaint_id.in_(complaint_ids))
                .order_by(ComplaintReply.created_at.asc())
            ).scalars().all()
            for r in rep_rows:
                if since_dt is not None:
                    rt = r.created_at
                    if rt is not None and rt.tzinfo is None:
                        rt = rt.replace(tzinfo=timezone.utc)
                    if rt is None or rt < since_dt:
                        continue
                replies_by_complaint.setdefault(r.complaint_id, []).append({
                    "id": r.id,
                    "author_kind": r.author_kind,
                    "author_label": r.author_label,
                    "content": r.content,
                    "created_at": _iso_utc(r.created_at),
                })

        items = []
        for c in complaints:
            replies = replies_by_complaint.get(c.id, [])
            # Skip complaints that have no replies AND were filtered by `since`.
            if since_dt is not None and not replies:
                continue
            items.append({
                "complaint_id": c.id,
                "severity": c.severity,
                "topic": c.topic,
                "status": c.status,
                "escalated": bool(c.escalated),
                "assigned_to": c.assigned_to,
                "complaint_created_at": _iso_utc(c.created_at),
                "replies": replies,
            })
    return {"user_id": user_id, "items": items}


@app.get("/audit")
def list_audit(
    tenant_id: str | None = None,
    path: str | None = None,
    event_type: str | None = None,
    user_id: str | None = None,
    since: str | None = None,   # ISO-8601 lower bound
    limit: int = 100,
    auth: AuthContext = Depends(require_api_key),
):
    """List audit records, newest first. Filterable by tenant/path/event/user/time.

    Meant for operators -- anyone with an API key for the relevant tenant can
    pull their own audit trail. When ``REQUIRE_API_KEY`` is on, callers are
    scoped to ``auth.tenant_id`` (overriding any ``tenant_id`` query param).
    """
    from datetime import datetime, timezone
    from sqlalchemy import select
    from rag.db.base import SessionLocal
    from rag.db.models import AuditLog

    effective_tenant = tenant_id
    if auth.is_authenticated:
        # Don't let an authed caller read another tenant's audit.
        effective_tenant = auth.tenant_id

    stmt = select(AuditLog)
    if effective_tenant:
        stmt = stmt.where(AuditLog.tenant_id == effective_tenant)
    if path:
        stmt = stmt.where(AuditLog.path == path)
    if event_type:
        stmt = stmt.where(AuditLog.event_type == event_type)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if since:
        try:
            dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            stmt = stmt.where(AuditLog.created_at >= dt)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"bad 'since' value: {since!r}")
    stmt = stmt.order_by(AuditLog.created_at.desc()).limit(max(1, min(int(limit), 500)))

    with SessionLocal() as s:
        rows = s.execute(stmt).scalars().all()
        items = [
            {
                "id": r.id,
                "trace_id": r.trace_id,
                "tenant_id": r.tenant_id,
                "user_id": r.user_id,
                "event_type": r.event_type,
                "method": r.method,
                "path": r.path,
                "status_code": r.status_code,
                "latency_ms": r.latency_ms,
                "query_hash": r.query_hash,
                "answer_hash": r.answer_hash,
                "error": r.error,
                "extra": r.extra or {},
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    return {"count": len(items), "items": items}


@app.post("/search")
def search(req: SearchRequest, auth: AuthContext = Depends(require_api_key)):
    try:
        kb = kb_mod.get_kb(req.kb_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    q = normalize_query(req.query)
    if req.use_rewrite:
        q = rewrite_query(q, conversation=None)
    hits = retrieve(
        q,
        kb.root,
        rerank=req.use_rerank,
        final_top_k=req.top_k or settings.final_top_k,
    )
    return {
        "query": req.query,
        "rewritten_query": q,
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "score": h.score,
                "title": h.title,
                "section_path": h.section_path,
                "source_id": h.source_id,
                "source_path": h.source_path,
                "retrieval_source": h.retrieval_source,
                "text": h.text,
            }
            for h in hits
        ],
    }


