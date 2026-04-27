"""add user_preferences (4th-layer agent memory)

Revision ID: user_preferences_01
Revises: sessions_modela_01
Create Date: 2026-04-26 14:00:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "user_preferences_01"
down_revision: Union[str, Sequence[str], None] = "sessions_modela_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="user"),
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
    with op.batch_alter_table("user_preferences", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_user_preferences_user_id"), ["user_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("user_preferences", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_user_preferences_user_id"))
    op.drop_table("user_preferences")
