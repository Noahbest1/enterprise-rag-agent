"""add user_addresses

Revision ID: tier2addr01
Revises: tier2cpl01
Create Date: 2026-04-23 19:00:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "tier2addr01"
down_revision: Union[str, Sequence[str], None] = "tier2cpl01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_addresses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False, server_default="家"),
        sa.Column("recipient", sa.String(length=64), nullable=False),
        sa.Column("phone", sa.String(length=32), nullable=False),
        sa.Column("province", sa.String(length=32), nullable=True),
        sa.Column("city", sa.String(length=32), nullable=True),
        sa.Column("district", sa.String(length=32), nullable=True),
        sa.Column("line1", sa.String(length=256), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("0")),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("user_addresses", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_user_addresses_user_id"), ["user_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("user_addresses", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_user_addresses_user_id"))
    op.drop_table("user_addresses")
