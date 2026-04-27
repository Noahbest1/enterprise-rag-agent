"""add audit_logs

Revision ID: extauditlog01
Revises: ph5apikeys01
Create Date: 2026-04-23 17:30:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "extauditlog01"
down_revision: Union[str, Sequence[str], None] = "ph5apikeys01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("tenant_id", sa.String(length=64), nullable=True),
        sa.Column("api_key_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=48), nullable=False),
        sa.Column("method", sa.String(length=8), nullable=True),
        sa.Column("path", sa.String(length=256), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("query_hash", sa.String(length=16), nullable=True),
        sa.Column("answer_hash", sa.String(length=16), nullable=True),
        sa.Column("error", sa.String(length=512), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_audit_logs_trace_id"), ["trace_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_logs_tenant_id"), ["tenant_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_logs_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_logs_event_type"), ["event_type"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_logs_path"), ["path"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_logs_created_at"), ["created_at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_audit_logs_created_at"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_path"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_event_type"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_user_id"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_tenant_id"))
        batch_op.drop_index(batch_op.f("ix_audit_logs_trace_id"))
    op.drop_table("audit_logs")
