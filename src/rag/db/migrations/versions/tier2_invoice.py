"""add invoices

Revision ID: tier2inv01
Revises: extauditlog01
Create Date: 2026-04-23 18:00:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "tier2inv01"
down_revision: Union[str, Sequence[str], None] = "extauditlog01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "invoices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("order_id", sa.String(length=64), nullable=False),
        sa.Column("tenant", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("tax_id", sa.String(length=32), nullable=True),
        sa.Column("invoice_type", sa.String(length=16), nullable=False, server_default="electronic"),
        sa.Column("amount_cents", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="requested"),
        sa.Column("download_url", sa.String(length=512), nullable=True),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["order_id"], ["orders.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_invoices_order_id"), ["order_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_invoices_order_id"))
    op.drop_table("invoices")
