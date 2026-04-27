"""SSE streaming endpoints for the Multi-Agent.

Frontend usage:
    const es = new EventSource("/agent/chat?tenant=jd&q=我的iPhone想退货");
    es.addEventListener("plan", e => {...});
    es.addEventListener("specialist_done", e => {...});
    es.addEventListener("answer", e => {...});
    es.addEventListener("done", e => es.close());

Event types:
- ``agent_start``        : {"tenant","query","user_id"}
- ``plan``               : {"plan":[{step_id, agent, query, depends_on}]}
- ``specialist_start``   : {"agent":"order","step_id":2}
- ``specialist_done``    : {"agent":"order","step_id":2,"trace":{...}}
- ``answer``             : {"text":"...","citations":[...],"entities":{...}}
- ``done``               : {"total_latency_ms":...}
- ``error``              : {"detail":"..."}

Using LangGraph's native ``astream`` to emit events as each node completes;
no need to rebuild the graph or manually sequence specialists.
"""
from __future__ import annotations

import json
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.graph import build_graph
from agent.state import AgentState
from rag.answer.meta_answer import answer_chitchat, answer_meta
from rag.config import settings
from rag.query.intent import classify_intent
from rag.query.normalize import detect_language


router = APIRouter()


class AgentChatRequest(BaseModel):
    query: str
    tenant: str = "jd"  # jd | taobao
    user_id: Optional[str] = None
    thread_id: Optional[str] = None  # for multi-turn continuation (Day 4 checkpointer)
    history: Optional[list[dict]] = None


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_agent_turn(req: AgentChatRequest) -> AsyncGenerator[str, None]:
    if req.tenant not in ("jd", "taobao"):
        yield _sse("error", {"detail": f"unknown tenant: {req.tenant}"})
        return

    default_user = {"jd": "jd-demo-user", "taobao": "tb-demo-user"}[req.tenant]
    user_id = req.user_id or default_user

    t0 = time.perf_counter()

    yield _sse("agent_start", {"tenant": req.tenant, "query": req.query, "user_id": user_id})

    # Register/refresh the ChatSession metadata row. This powers the
    # session sidebar; it doesn't affect agent behaviour. Failures are
    # swallowed so a metadata write hiccup can't break the chat stream.
    thread_id_resolved = req.thread_id or f"anon-{user_id}"
    try:
        from agent.tools.sessions import touch_session
        touch_session(
            thread_id=thread_id_resolved,
            user_id=user_id,
            tenant=req.tenant,
            first_message=req.query,
        )
    except Exception:  # pragma: no cover -- never break the chat on metadata
        pass

    # Multi-turn: build a checkpointer-backed graph keyed on thread_id.
    # AsyncSqliteSaver persists state across turns -- entities (last_order_id /
    # last_item_title / etc.) survive, and ``messages`` accumulates via the
    # ``add`` reducer in AgentState. Without this, a follow-up like
    # "那能退吗?" lost context every turn.
    # async_mode=True is required because we call graph.astream below;
    # the sync SqliteSaver raises NotImplementedError in that path.
    graph = build_graph(with_checkpointer=True, async_mode=True)
    config = {"configurable": {"thread_id": req.thread_id or f"anon-{user_id}"}}

    new_user_msg = {"role": "user", "content": req.query}

    # Inspect prior state via the async checkpointer. First turn → snapshot empty.
    snapshot = None
    try:
        snapshot = await graph.aget_state(config)
    except Exception:
        snapshot = None
    has_prior = bool(
        snapshot
        and getattr(snapshot, "values", None)
        and snapshot.values.get("tenant")
    )

    # ── Intent routing on the Agent path (mirror of the RAG /answer/stream
    # path's intent router). Without this, meta-questions like "我之前问的
    #是哪个订单" route to the order specialist's default "list recent orders"
    # behavior and the LLM narrates "您之前咨询的是这些订单" — which is a lie
    # when the user hadn't actually asked about any specific order.
    if getattr(settings, "enable_intent_routing", True):
        prior_messages = (snapshot.values.get("messages") if has_prior else None) or []
        # Coerce LangGraph BaseMessage objects → simple dicts for meta handler.
        conversation: list[dict] = []
        for m in prior_messages:
            if isinstance(m, dict):
                conversation.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            else:
                # langchain_core.messages BaseMessage flavor
                role = getattr(m, "type", None) or "user"
                role = "assistant" if role == "ai" else ("user" if role == "human" else role)
                conversation.append({"role": role, "content": getattr(m, "content", "") or ""})
        verdict = classify_intent(req.query, has_conversation=bool(conversation))
        if verdict.intent == "meta":
            # Even with empty conversation, short-circuit and let
            # answer_meta return its polite "no prior turns" fallback.
            # Falling back to the planner here causes the order specialist
            # to list all of the user's orders and the LLM hallucinates
            # "you previously asked about these" — a real bug we hit.
            language = detect_language(req.query)
            text = await answer_meta(req.query, conversation, language)
            yield _sse("answer", {
                "text": text,
                "citations": [],
                "entities": (snapshot.values.get("entities") if has_prior else {}) or {},
                "intent": "meta",
                "intent_matched": verdict.matched,
            })
            yield _sse("done", {"total_latency_ms": int((time.perf_counter() - t0) * 1000)})
            return
        if verdict.intent == "chitchat":
            language = detect_language(req.query)
            text = await answer_chitchat(req.query, language)
            yield _sse("answer", {
                "text": text,
                "citations": [],
                "entities": (snapshot.values.get("entities") if has_prior else {}) or {},
                "intent": "chitchat",
                "intent_matched": verdict.matched,
            })
            yield _sse("done", {"total_latency_ms": int((time.perf_counter() - t0) * 1000)})
            return

    if has_prior:
        # Continuation: send a minimal delta. The ``add`` reducer on
        # ``messages`` appends instead of overwriting, so prior turns stay
        # in state. Per-turn fields (plan / current_step / step_results)
        # are explicitly reset so a stale plan from the prior turn doesn't
        # leak into our SSE stream before planner overwrites it.
        initial_state: AgentState = {
            "messages": [new_user_msg],
            "plan": [],
            "current_step": -1,
            "step_results": {},
        }
        # running_state mirrors the checkpoint so SSE events see prior entities.
        running_state: dict = dict(snapshot.values)
        running_state["messages"] = list(running_state.get("messages") or []) + [new_user_msg]
        running_state["plan"] = []
        running_state["current_step"] = -1
        running_state["step_results"] = {}
    else:
        # First turn: full initialise, including any history the caller
        # explicitly passed (legacy path; modern callers rely on the
        # checkpointer instead of round-tripping history through the wire).
        initial_state = {
            "tenant": req.tenant,
            "user_id": user_id,
            "messages": list(req.history or []) + [new_user_msg],
            "entities": {},
            "step_results": {},
            "trace": [],
        }
        running_state = {
            "tenant": req.tenant,
            "user_id": user_id,
            "messages": list(req.history or []) + [new_user_msg],
            "entities": {},
            "step_results": {},
            "trace": [],
        }

    # Track which step is currently running so we can emit specialist_start
    # events as soon as coordinator picks one.
    last_step_seen: Optional[int] = None
    specialist_names = {"product_qa", "policy_qa", "order", "logistics", "aftersale", "recommend", "invoice", "complaint", "account"}

    # LangGraph's stream_mode="updates" only sends the node-scoped patch, not
    # the full state. We mirror the state here by merging each patch so later
    # events (e.g. ``advance`` reading the plan written by ``planner``) work
    # correctly. Without this, plan stays [] in the ``advance`` branch below
    # and we never emit ``specialist_start`` -- the front-end then only sees
    # ``plan`` + ``answer`` and looks indistinguishable from plain RAG.

    def _merge(patch: dict) -> None:
        if not isinstance(patch, dict):
            return
        for k, v in patch.items():
            if k == "trace" and isinstance(v, list):
                running_state["trace"] = list(running_state.get("trace") or []) + v
            elif k == "step_results" and isinstance(v, dict):
                merged = dict(running_state.get("step_results") or {})
                merged.update(v)
                running_state["step_results"] = merged
            else:
                running_state[k] = v

    try:
        async for update in graph.astream(initial_state, stream_mode="updates", config=config):
            # update = {"<node_name>": <state patch>}
            for node_name, patch in update.items():
                _merge(patch)

                if node_name == "planner":
                    plan = running_state.get("plan") or []
                    yield _sse("plan", {"plan": plan})

                elif node_name == "advance":
                    # About to dispatch to a specialist. ``current_step`` was
                    # just written by advance; ``plan`` was written earlier
                    # by planner and is now visible via running_state.
                    idx = running_state.get("current_step", -1)
                    plan = running_state.get("plan") or []
                    if idx is not None and 0 <= idx < len(plan):
                        step = plan[idx]
                        yield _sse("specialist_start", {
                            "agent": step.get("agent"),
                            "step_id": step.get("step_id"),
                            "query": step.get("query"),
                        })
                        last_step_seen = step.get("step_id")

                elif node_name in specialist_names:
                    # This specialist just finished; surface its specific
                    # step result (not the whole cumulative trace).
                    sr = running_state.get("step_results") or {}
                    this_step_output = None
                    if last_step_seen is not None:
                        this_step_output = sr.get(last_step_seen)
                    yield _sse("specialist_done", {
                        "agent": node_name,
                        "step_id": last_step_seen,
                        "output": this_step_output,
                    })

                elif node_name == "summary":
                    final = running_state.get("final_answer") or patch.get("final_answer", "")
                    messages = running_state.get("messages") or []
                    citations = []
                    if messages and isinstance(messages[-1], dict):
                        citations = messages[-1].get("citations") or []
                    yield _sse("answer", {
                        "text": final,
                        "citations": citations,
                        "entities": running_state.get("entities") or {},
                    })

        yield _sse("done", {"total_latency_ms": int((time.perf_counter() - t0) * 1000)})

    except Exception as e:
        yield _sse("error", {"detail": f"{type(e).__name__}: {e}"})


@router.post("/agent/chat")
async def agent_chat_stream(req: AgentChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query required")

    return StreamingResponse(
        _stream_agent_turn(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/agent/chat")
async def agent_chat_stream_get(query: str, tenant: str = "jd", user_id: str | None = None):
    """GET variant so EventSource clients (which are GET-only) can connect."""
    req = AgentChatRequest(query=query, tenant=tenant, user_id=user_id)
    return await agent_chat_stream(req)
