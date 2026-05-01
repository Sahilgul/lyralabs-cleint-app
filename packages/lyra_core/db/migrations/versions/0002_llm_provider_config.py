"""LLM provider config + per-tier model assignment

Adds two tables that back the super-admin model-switcher:

  llm_providers          -- per-vendor credentials (api key encrypted with the
                            platform key, optional api_base override, extra
                            config JSONB).
  llm_model_assignments  -- which (provider, model_id) is active per tier.

Both tables are empty after this migration runs; the LLM router falls back
to LLM_PRIMARY_MODEL / LLM_CHEAP_MODEL env vars until the operator
configures providers via the admin UI. So this migration is safe to deploy
ahead of any UI changes.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-01
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_providers",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("provider_key", sa.String(32), nullable=False, unique=True),
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("api_base", sa.Text(), nullable=True),
        sa.Column(
            "extra_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("last_tested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_test_status", sa.String(32), nullable=True),
        sa.Column("last_test_error", sa.Text(), nullable=True),
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
        sa.Column("updated_by_email", sa.String(256), nullable=True),
    )
    op.create_index("ix_llm_providers_provider_key", "llm_providers", ["provider_key"])

    op.create_table(
        "llm_model_assignments",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("tier", sa.String(32), nullable=False),
        sa.Column("provider_key", sa.String(32), nullable=False),
        sa.Column("model_id", sa.String(128), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("updated_by_email", sa.String(256), nullable=True),
        sa.UniqueConstraint("tier", name="uq_llm_assignment_tier"),
    )
    op.create_index("ix_llm_model_assignments_tier", "llm_model_assignments", ["tier"])
    op.create_index(
        "ix_llm_model_assignments_provider_key",
        "llm_model_assignments",
        ["provider_key"],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_model_assignments_provider_key", table_name="llm_model_assignments")
    op.drop_index("ix_llm_model_assignments_tier", table_name="llm_model_assignments")
    op.drop_table("llm_model_assignments")
    op.drop_index("ix_llm_providers_provider_key", table_name="llm_providers")
    op.drop_table("llm_providers")
