"""Client + Campaign schema; client_id on integrations, jobs, audit_events.

Adds:
  - clients table  (id, tenant_id, name, slug, primary_slack_channel_id, status, timestamps)
  - campaigns table (id, tenant_id, client_id, name, kind, status, spec, state_data, timestamps)
  - client_id column (nullable FK → clients) on:
      integration_connections, jobs, audit_events
  - Replaces uq_integration_per_account (tenant+provider+acct) with
    (tenant+client+provider+acct) to allow the same provider to have separate
    credentials per client.
  - Backfills one synthetic "agency_internal" Client per existing Tenant,
    then points all existing rows at it.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. clients table
    # ------------------------------------------------------------------
    op.create_table(
        "clients",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("primary_slack_channel_id", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
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
    op.create_index("ix_clients_tenant_id", "clients", ["tenant_id"])
    op.create_unique_constraint("uq_client_slug_per_tenant", "clients", ["tenant_id", "slug"])
    op.create_unique_constraint(
        "uq_client_slack_channel",
        "clients",
        ["tenant_id", "primary_slack_channel_id"],
    )

    # ------------------------------------------------------------------
    # 2. campaigns table (schema lock-in; no code uses it yet)
    # ------------------------------------------------------------------
    op.create_table(
        "campaigns",
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
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="draft"),
        sa.Column(
            "spec",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "state_data",
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
    op.create_index("ix_campaigns_tenant_id", "campaigns", ["tenant_id"])
    op.create_index("ix_campaigns_client_id", "campaigns", ["client_id"])

    # ------------------------------------------------------------------
    # 3. Add client_id to integration_connections
    # ------------------------------------------------------------------
    op.add_column(
        "integration_connections",
        sa.Column(
            "client_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("clients.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_integration_client_id", "integration_connections", ["client_id"])
    # Replace old unique constraint (tenant+provider+acct) with (tenant+client+provider+acct)
    op.drop_constraint("uq_integration_per_account", "integration_connections", type_="unique")
    op.create_unique_constraint(
        "uq_integration_per_account",
        "integration_connections",
        ["tenant_id", "client_id", "provider", "external_account_id"],
    )

    # ------------------------------------------------------------------
    # 4. Add client_id to jobs
    # ------------------------------------------------------------------
    op.add_column(
        "jobs",
        sa.Column(
            "client_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("clients.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_jobs_client_id", "jobs", ["client_id"])
    op.create_index("ix_jobs_tenant_client", "jobs", ["tenant_id", "client_id"])

    # ------------------------------------------------------------------
    # 5. Add client_id to audit_events
    # ------------------------------------------------------------------
    op.add_column(
        "audit_events",
        sa.Column(
            "client_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("clients.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_audit_client_id", "audit_events", ["client_id"])
    op.create_index("ix_audit_tenant_client_ts", "audit_events", ["tenant_id", "client_id", "ts"])

    # ------------------------------------------------------------------
    # 6. Data backfill
    #    - One synthetic "agency_internal" Client per existing Tenant
    #    - Point all existing FK rows at it
    # ------------------------------------------------------------------
    op.execute("""
        INSERT INTO clients (id, tenant_id, name, slug, status, created_at, updated_at)
        SELECT
            gen_random_uuid(),
            id,
            name || ' (internal)',
            'agency_internal',
            'active',
            now(),
            now()
        FROM tenants
        ON CONFLICT DO NOTHING
    """)

    op.execute("""
        UPDATE integration_connections ic
        SET client_id = (
            SELECT c.id FROM clients c
            WHERE c.tenant_id = ic.tenant_id AND c.slug = 'agency_internal'
            LIMIT 1
        )
        WHERE ic.client_id IS NULL
    """)

    op.execute("""
        UPDATE jobs j
        SET client_id = (
            SELECT c.id FROM clients c
            WHERE c.tenant_id = j.tenant_id AND c.slug = 'agency_internal'
            LIMIT 1
        )
        WHERE j.client_id IS NULL
    """)

    op.execute("""
        UPDATE audit_events a
        SET client_id = (
            SELECT c.id FROM clients c
            WHERE c.tenant_id = a.tenant_id AND c.slug = 'agency_internal'
            LIMIT 1
        )
        WHERE a.client_id IS NULL
    """)


def downgrade() -> None:
    # Reverse order
    op.drop_index("ix_audit_tenant_client_ts", table_name="audit_events")
    op.drop_index("ix_audit_client_id", table_name="audit_events")
    op.drop_column("audit_events", "client_id")

    op.drop_index("ix_jobs_tenant_client", table_name="jobs")
    op.drop_index("ix_jobs_client_id", table_name="jobs")
    op.drop_column("jobs", "client_id")

    op.drop_index("ix_integration_client_id", table_name="integration_connections")
    op.drop_constraint("uq_integration_per_account", "integration_connections", type_="unique")
    op.create_unique_constraint(
        "uq_integration_per_account",
        "integration_connections",
        ["tenant_id", "provider", "external_account_id"],
    )
    op.drop_column("integration_connections", "client_id")

    op.drop_index("ix_campaigns_client_id", table_name="campaigns")
    op.drop_index("ix_campaigns_tenant_id", table_name="campaigns")
    op.drop_table("campaigns")

    op.drop_constraint("uq_client_slack_channel", "clients", type_="unique")
    op.drop_constraint("uq_client_slug_per_tenant", "clients", type_="unique")
    op.drop_index("ix_clients_tenant_id", table_name="clients")
    op.drop_table("clients")
