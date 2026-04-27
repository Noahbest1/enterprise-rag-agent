"""ORM models. Keep columns small and explicit; don't leak SQLAlchemy objects
across module boundaries -- callers should receive plain dicts or dataclasses.

Schema rationale
----------------
- users / orders / order_items -- the e-commerce agent's Order specialist
  queries these. Mock data but real schema, so a future real DB swap is
  just a data migration.
- feedback -- user thumbs up/down with optional free-text. Fuel for the
  eval regression pipeline (Day 9).
- kb_metadata -- one row per registered KB. Lets us list / introspect KBs
  without scanning the filesystem.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant: Mapped[str] = mapped_column(String(16))  # "jd" | "taobao"
    display_name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    orders: Mapped[list["Order"]] = relationship(back_populates="user")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    tenant: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32))  # placed / paid / shipped / delivered / cancelled / refunded
    total_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    tracking_no: Mapped[str | None] = mapped_column(String(64), nullable=True)
    carrier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[User] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"))
    sku: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(256))
    qty: Mapped[int] = mapped_column(Integer, default=1)
    unit_price_cents: Mapped[int] = mapped_column(Integer)

    order: Mapped[Order] = relationship(back_populates="items")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    verdict: Mapped[str] = mapped_column(String(16))  # "up" | "down"
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReturnRequest(Base):
    """A user's return/refund/exchange request against a specific order.

    Created by the AfterSale specialist when a user asks for a return. Mock
    downstream: in production this would kick off a workflow queue; here
    we just persist the record so future conversation turns can recall it.
    """
    __tablename__ = "return_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    tenant: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16))  # "refund" | "return" | "exchange" | "price_protect"
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | approved | rejected | completed
    refund_cents: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class UserAddress(Base):
    """User's shipping addresses.

    Sensitive PII by definition -- recipient name, phone, street. Not redacted
    here (user's own data in their own account), but:
    - DO NOT bulk-export. The `/audit` admin endpoint's tenant scoping still
      applies; it won't let one tenant read another tenant's addresses.
    - Phone is stored raw (not hashed). Changing phone goes through the
      Account specialist's mock SMS-verification step, never a direct write.
    """
    __tablename__ = "user_addresses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True,
    )
    label: Mapped[str] = mapped_column(String(32), default="家")  # 家 / 公司 / 其他
    recipient: Mapped[str] = mapped_column(String(64))
    phone: Mapped[str] = mapped_column(String(32))
    province: Mapped[str | None] = mapped_column(String(32), nullable=True)
    city: Mapped[str | None] = mapped_column(String(32), nullable=True)
    district: Mapped[str | None] = mapped_column(String(32), nullable=True)
    line1: Mapped[str] = mapped_column(String(256))  # 详细街道 + 门牌
    is_default: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Complaint(Base):
    """Customer complaint ticket.

    Rows are written by the Complaint specialist when a user expresses
    dissatisfaction. High-severity rows are auto-escalated (``status =
    escalated`` + ``assigned_to`` set) so the UI / downstream ops can
    treat them as human-needed. Low / medium rows stay in ``open`` and
    follow the SLA timers (``sla_due_at``).

    The raw user content is NOT stored, only a sha256[:16] hash --
    consistent with the audit-log policy. The Summary node can still
    quote the content from state.messages; it just doesn't live in the
    long-lived complaint record.
    """
    __tablename__ = "complaints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    tenant: Mapped[str] = mapped_column(String(16))
    # The ChatSession that filed this complaint, so the user's session
    # sidebar can show "tickets created in this session". Nullable for
    # backwards compatibility with rows created before sessions existed.
    thread_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    order_id: Mapped[str | None] = mapped_column(
        ForeignKey("orders.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    topic: Mapped[str] = mapped_column(String(16))  # delivery | quality | service | refund | price | other
    severity: Mapped[str] = mapped_column(String(8))  # low | medium | high
    content_hash: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | escalated | resolved | closed
    escalated: Mapped[bool] = mapped_column(default=False)
    assigned_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sla_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class UserPreference(Base):
    """4th layer of Agent memory: durable, cross-session user preferences.

    The first three layers (in-turn ``messages``, in-turn ``entities``,
    cross-session LangGraph checkpoint) all live inside one conversation
    thread. UserPreference is the **persistent layer above sessions** —
    "user u1 prefers JD courier for shipping" survives session deletion,
    new logins, and the user's first question on a brand-new session.

    Schema is intentionally a key-value bag (not separate columns per
    preference) so we can add new preference types without migrations.
    """
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True,
    )
    key: Mapped[str] = mapped_column(String(64))
    value: Mapped[str] = mapped_column(Text)
    # Source attribution: 'user' = explicit ("把 JD 设为默认快递"),
    # 'inferred' = from observation (3+ orders all chose JD),
    # 'admin' = set by CS rep on user's behalf.
    source: Mapped[str] = mapped_column(String(16), default="user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ChatSession(Base):
    """One row per agent chat thread — the ChatGPT-style "session" concept.

    Maps the LangGraph ``thread_id`` (already used as the AsyncSqliteSaver
    key) back to a ``user_id`` + tenant + display title, so the frontend
    can list a user's past sessions and switch between them. Deleting a
    row here also clears the corresponding LangGraph checkpoint
    (separate concern, see scripts / endpoint).

    The actual conversation state (messages / entities / step_results)
    still lives in ``data/langgraph.sqlite`` keyed on the same thread_id;
    this table is metadata only.
    """
    __tablename__ = "chat_sessions"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    tenant: Mapped[str] = mapped_column(String(16))
    # Auto-populated from the first user message (truncated 40 chars). User
    # can rename via PATCH /sessions/{id}.
    title: Mapped[str] = mapped_column(String(80), default="(新会话)")
    first_msg_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_msg_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ComplaintReply(Base):
    """A single admin (or system) reply on a complaint thread.

    Admin-authored content so it IS stored raw (unlike complaints.content_hash).
    Each insert publishes an event on the in-process user-events bus so the
    target user's open AgentChat SSE connection injects a system bubble.
    """
    __tablename__ = "complaint_replies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    complaint_id: Mapped[int] = mapped_column(
        ForeignKey("complaints.id", ondelete="CASCADE"), index=True,
    )
    author_kind: Mapped[str] = mapped_column(String(16), default="admin")  # admin | system
    author_label: Mapped[str] = mapped_column(String(64))  # e.g. "jd-cs-senior-A"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Invoice(Base):
    """Invoice issued for an order (一张订单对应一张电子发票).

    A typical customer-service flow:
        1. user: "帮我开 JD20260420456 的发票"
        2. InvoiceAgent looks up: if status == "issued", return download_url;
           else insert a row in "requested" and return an ETA.
        3. A separate async worker flips "requested" → "issued" (out of scope
           for this demo; we mock it in seeds).
    """
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    tenant: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(256))  # 抬头: "个人" or a company name
    tax_id: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 税号,企业发票才有
    invoice_type: Mapped[str] = mapped_column(String(16), default="electronic")  # electronic | paper
    amount_cents: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="requested")  # requested | issued | cancelled
    download_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KBDocument(Base):
    """Registry of documents sourced from connectors.

    One row per logical document (one HTML page / one PDF / one MD file).
    Enables incremental ingest: compare connector output to these rows by
    content_hash; add / update / delete accordingly.

    ``source_id`` is the same key the chunk-level schema already uses
    (see ingest.pipeline._source_id), so deleting a row's chunks from both
    BM25 and vector backends is a single ``WHERE source_id = ?``.
    """
    __tablename__ = "kb_documents"

    source_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(64), index=True)
    connector: Mapped[str] = mapped_column(String(32))
    source_uri: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ApiKey(Base):
    """API credential issued to a single tenant.

    We store only the sha256 hash of the key. The raw key is shown once at
    creation time (`scripts/create_api_key.py`). Lookup is O(1) on the hash
    since each request passes the raw key and we hash + query by hash.
    """
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Optional per-key scopes; empty string means "all".
    scopes: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    disabled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AuditLog(Base):
    """Immutable request-level audit trail.

    Written by the HTTP middleware (one row per request) and/or by
    business code for sensitive events (return_request created, API key
    issued, GDPR delete). Never stores the raw query/answer body -- only
    their sha256 prefix so operators can confirm "this trace_id asked that
    question" without leaking PII into the audit stream.

    Queries that need full text recovery must join back to application
    logs via ``trace_id``. That preserves the audit table as a slim,
    searchable spine.
    """
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    api_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    event_type: Mapped[str] = mapped_column(String(48), index=True)  # http_request / return_request_created / gdpr_delete / api_key_issued / ...
    method: Mapped[str | None] = mapped_column(String(8), nullable=True)
    path: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    query_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    answer_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error: Mapped[str | None] = mapped_column(String(512), nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True,
    )


class KBMetadata(Base):
    __tablename__ = "kb_metadata"

    kb_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant: Mapped[str | None] = mapped_column(String(16), nullable=True)
    vector_backend: Mapped[str] = mapped_column(String(16), default="faiss")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
