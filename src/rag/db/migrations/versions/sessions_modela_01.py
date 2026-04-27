"""add chat_sessions table + complaints.thread_id column

Revision ID: sessions_modela_01
Revises: stepb_replies_01
Create Date: 2026-04-26 12:00:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "sessions_modela_01"
down_revision: Union[str, Sequence[str], None] = "stepb_replies_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=80), nullable=False, server_default="(新会话)"),
        sa.Column(
            "first_msg_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "last_msg_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_chat_sessions_user_id"), ["user_id"], unique=False)

    # complaints.thread_id — nullable, indexed for "tickets in this session" queries.
    with op.batch_alter_table("complaints", schema=None) as batch_op:
        batch_op.add_column(sa.Column("thread_id", sa.String(length=64), nullable=True))
        batch_op.create_index(batch_op.f("ix_complaints_thread_id"), ["thread_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("complaints", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_complaints_thread_id"))
        batch_op.drop_column("thread_id")
    with op.batch_alter_table("chat_sessions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_chat_sessions_user_id"))
    op.drop_table("chat_sessions")
