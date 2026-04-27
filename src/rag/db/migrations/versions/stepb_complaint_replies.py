"""add complaint_replies

Revision ID: stepb_replies_01
Revises: tier2addr01
Create Date: 2026-04-24 14:00:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "stepb_replies_01"
down_revision: Union[str, Sequence[str], None] = "tier2addr01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "complaint_replies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("complaint_id", sa.Integer(), nullable=False),
        sa.Column("author_kind", sa.String(length=16), nullable=False, server_default="admin"),
        sa.Column("author_label", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["complaint_id"], ["complaints.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("complaint_replies", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_complaint_replies_complaint_id"),
            ["complaint_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("complaint_replies", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_complaint_replies_complaint_id"))
    op.drop_table("complaint_replies")
