"""Living Artifact: workspace_artifacts table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_artifacts",
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
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column(
            "body",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
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
    op.create_index("ix_workspace_artifacts_tenant_id", "workspace_artifacts", ["tenant_id"])
    op.create_index("ix_workspace_artifacts_client_id", "workspace_artifacts", ["client_id"])
    op.create_index("ix_workspace_artifacts_thread_id", "workspace_artifacts", ["thread_id"])
    op.create_unique_constraint(
        "uq_artifact_per_thread",
        "workspace_artifacts",
        ["tenant_id", "client_id", "thread_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_artifact_per_thread", "workspace_artifacts", type_="unique")
    op.drop_index("ix_workspace_artifacts_thread_id", table_name="workspace_artifacts")
    op.drop_index("ix_workspace_artifacts_client_id", table_name="workspace_artifacts")
    op.drop_index("ix_workspace_artifacts_tenant_id", table_name="workspace_artifacts")
    op.drop_table("workspace_artifacts")
