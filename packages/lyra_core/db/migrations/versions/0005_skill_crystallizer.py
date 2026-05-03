"""Skill Crystallizer: skills table.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "skills",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "client_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("clients.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "tool_sequence",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("promoted_by", sa.String(128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_skills_tenant_id", "skills", ["tenant_id"])
    op.create_index("ix_skills_client_id", "skills", ["client_id"])
    op.create_unique_constraint("uq_skill_per_client", "skills", ["tenant_id", "client_id", "slug"])


def downgrade() -> None:
    op.drop_constraint("uq_skill_per_client", "skills", type_="unique")
    op.drop_index("ix_skills_client_id", table_name="skills")
    op.drop_index("ix_skills_tenant_id", table_name="skills")
    op.drop_table("skills")
