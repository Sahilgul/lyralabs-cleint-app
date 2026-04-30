"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-30 00:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("external_team_id", sa.String(64), nullable=False, unique=True),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("plan", sa.String(32), nullable=False, server_default="trial"),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "trial_credit_remaining_usd", sa.Float(), nullable=False, server_default="100.0"
        ),
        sa.Column(
            "settings", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("stripe_customer_id", sa.String(64), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(64), nullable=True),
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
    op.create_index("ix_tenants_external_team_id", "tenants", ["external_team_id"])

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_user_id", sa.String(128), nullable=False),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("display_name", sa.String(256), nullable=True),
        sa.Column("email", sa.String(256), nullable=True),
        sa.Column("role", sa.String(32), nullable=False, server_default="member"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "tenant_id", "external_user_id", "channel", name="uq_user_per_tenant"
        ),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])

    op.create_table(
        "slack_installations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("team_id", sa.String(64), nullable=True),
        sa.Column("team_name", sa.String(256), nullable=True),
        sa.Column("enterprise_id", sa.String(64), nullable=True),
        sa.Column("enterprise_name", sa.String(256), nullable=True),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("bot_token_encrypted", sa.Text(), nullable=True),
        sa.Column("bot_id", sa.String(64), nullable=True),
        sa.Column("bot_user_id", sa.String(64), nullable=True),
        sa.Column("bot_scopes", sa.Text(), nullable=True),
        sa.Column("bot_refresh_token_encrypted", sa.Text(), nullable=True),
        sa.Column("bot_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_token_encrypted", sa.Text(), nullable=True),
        sa.Column("user_scopes", sa.Text(), nullable=True),
        sa.Column("incoming_webhook_url", sa.Text(), nullable=True),
        sa.Column("incoming_webhook_channel", sa.String(128), nullable=True),
        sa.Column("incoming_webhook_channel_id", sa.String(64), nullable=True),
        sa.Column("incoming_webhook_configuration_url", sa.Text(), nullable=True),
        sa.Column(
            "is_enterprise_install", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("token_type", sa.String(32), nullable=True),
        sa.Column(
            "installed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_slack_install_team_enterprise", "slack_installations", ["team_id", "enterprise_id"])
    op.create_index("ix_slack_installations_tenant_id", "slack_installations", ["tenant_id"])

    op.create_table(
        "integration_connections",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("external_account_id", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(256), nullable=True),
        sa.Column("scopes", sa.Text(), nullable=False, server_default=""),
        sa.Column("access_token_encrypted", sa.Text(), nullable=False),
        sa.Column("refresh_token_encrypted", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
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
        sa.UniqueConstraint(
            "tenant_id", "provider", "external_account_id", name="uq_integration_per_account"
        ),
    )
    op.create_index("ix_integration_connections_tenant_id", "integration_connections", ["tenant_id"])
    op.create_index("ix_integration_connections_provider", "integration_connections", ["provider"])

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.String(128), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=True),
        sa.Column("channel_id", sa.String(128), nullable=True),
        sa.Column("parent_message_ts", sa.String(64), nullable=True),
        sa.Column("user_request", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("plan_json", postgresql.JSONB(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column(
            "artifact_urls", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"),
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
    op.create_index("ix_jobs_tenant_id", "jobs", ["tenant_id"])
    op.create_index("ix_jobs_thread_id", "jobs", ["thread_id"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("actor_user_id", sa.String(128), nullable=True),
        sa.Column("job_id", sa.String(64), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("tool_name", sa.String(128), nullable=True),
        sa.Column("args_hash", sa.String(64), nullable=True),
        sa.Column("raw_args", postgresql.JSONB(), nullable=True),
        sa.Column("result_status", sa.String(32), nullable=False, server_default="ok"),
        sa.Column("model_used", sa.String(128), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "extra", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_audit_events_tenant_id", "audit_events", ["tenant_id"])
    op.create_index("ix_audit_events_job_id", "audit_events", ["job_id"])
    op.create_index("ix_audit_events_ts", "audit_events", ["ts"])
    op.create_index("ix_audit_tenant_ts", "audit_events", ["tenant_id", "ts"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("jobs")
    op.drop_table("integration_connections")
    op.drop_table("slack_installations")
    op.drop_table("users")
    op.drop_table("tenants")
