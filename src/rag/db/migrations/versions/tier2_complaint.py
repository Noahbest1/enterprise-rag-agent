"""add complaints

Revision ID: tier2cpl01
Revises: tier2inv01
Create Date: 2026-04-23 18:30:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "tier2cpl01"
down_revision: Union[str, Sequence[str], None] = "tier2inv01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "complaints",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("tenant", sa.String(length=16), nullable=False),
        sa.Column("order_id", sa.String(length=64), nullable=True),
        sa.Column("topic", sa.String(length=16), nullable=False),
        sa.Column("severity", sa.String(length=8), nullable=False),
        sa.Column("content_hash", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
        sa.Column("escalated", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("assigned_to", sa.String(length=64), nullable=True),
        sa.Column("sla_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("complaints", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_complaints_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_complaints_order_id"), ["order_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("complaints", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_complaints_order_id"))
        batch_op.drop_index(batch_op.f("ix_complaints_user_id"))
    op.drop_table("complaints")
