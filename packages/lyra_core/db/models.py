"""SQLAlchemy 2.0 ORM models.

Tenant = Slack workspace (or Teams tenant). All other tables FK to it.
OAuth tokens are stored encrypted (see common/crypto.py).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB}


# -----------------------------------------------------------------------------
# Tenancy
# -----------------------------------------------------------------------------


class Tenant(Base):
    """One per Slack workspace or Teams tenant.

    `external_team_id` is the Slack `team_id` (T0123) or Teams tenant id.
    """

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    external_team_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    channel: Mapped[str] = mapped_column(String(16))  # "slack" | "teams"
    name: Mapped[str] = mapped_column(String(256))
    plan: Mapped[str] = mapped_column(
        String(32), default="trial"
    )  # trial|team|enterprise|cancelled
    status: Mapped[str] = mapped_column(String(32), default="active")  # active|past_due|cancelled
    trial_credit_remaining_usd: Mapped[float] = mapped_column(Float, default=100.0)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    users: Mapped[list[User]] = relationship(back_populates="tenant", cascade="all, delete-orphan")
    integrations: Mapped[list[IntegrationConnection]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )
    clients: Mapped[list[Client]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )


class User(Base):
    """A human in a tenant (Slack user, Teams user, or admin-panel user).

    For Slack/Teams users, `external_user_id` is the platform user id.
    For admin-panel users, channel='admin' and external_user_id is the auth provider id.
    """

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_user_id", "channel", name="uq_user_per_tenant"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    external_user_id: Mapped[str] = mapped_column(String(128))
    channel: Mapped[str] = mapped_column(String(16))  # slack|teams|admin
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="member")  # member|admin|owner
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    tenant: Mapped[Tenant] = relationship(back_populates="users")


class AdminUser(Base):
    """Admin-panel user account (email + bcrypt password).

    Separate from the Slack/Teams `User` table — these are workspace admins
    who sign in to the admin UI.
    """

    __tablename__ = "admin_users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="owner")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# -----------------------------------------------------------------------------
# Channel installations
# -----------------------------------------------------------------------------


class SlackInstallation(Base):
    """Persistent Slack OAuth install (used by slack_bolt InstallationStore).

    We mirror the fields slack_bolt expects so we can implement a custom
    InstallationStore backed by Postgres.
    """

    __tablename__ = "slack_installations"
    __table_args__ = (Index("ix_slack_install_team_enterprise", "team_id", "enterprise_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    team_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    team_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    enterprise_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enterprise_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bot_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    bot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bot_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bot_scopes: Mapped[str | None] = mapped_column(Text, nullable=True)
    bot_refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    bot_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    user_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_scopes: Mapped[str | None] = mapped_column(Text, nullable=True)
    incoming_webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    incoming_webhook_channel: Mapped[str | None] = mapped_column(String(128), nullable=True)
    incoming_webhook_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    incoming_webhook_configuration_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_enterprise_install: Mapped[bool] = mapped_column(default=False)
    token_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    installed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# -----------------------------------------------------------------------------
# Clients and Campaigns
# -----------------------------------------------------------------------------


class Client(Base):
    """A customer or sub-account under a Tenant. For agencies, one row per client brand.

    GHL location IDs, Meta account IDs, etc. live in IntegrationConnection.external_account_id
    keyed by (tenant_id, client_id, provider) — not in a JSONB blob here.
    """

    __tablename__ = "clients"
    __table_args__ = (
        Index("ix_clients_tenant_id", "tenant_id"),
        UniqueConstraint("tenant_id", "slug", name="uq_client_slug_per_tenant"),
        # Primary O(1) lookup: Slack adapter resolves inbound channel_id → client_id
        UniqueConstraint("tenant_id", "primary_slack_channel_id", name="uq_client_slack_channel"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(Text)
    # slug is how users reference clients in Slack: "@ARLO pull deals for `acme`"
    slug: Mapped[str] = mapped_column(Text)
    primary_slack_channel_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active|paused|archived

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    tenant: Mapped[Tenant] = relationship(back_populates="clients")
    campaigns: Mapped[list[Campaign]] = relationship(
        back_populates="client", cascade="all, delete-orphan"
    )


class Campaign(Base):
    """A cross-tool marketing campaign owned by a Client.

    Schema lands now; agent code paths use it in v2. The spec/state_data JSONB
    fields are flexible enough to back any campaign state machine.
    """

    __tablename__ = "campaigns"
    __table_args__ = (
        Index("ix_campaigns_tenant_id", "tenant_id"),
        Index("ix_campaigns_client_id", "client_id"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(Text)
    kind: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # e.g. "email_sequence", "sms_blast"
    status: Mapped[str] = mapped_column(
        String(32), default="draft"
    )  # draft|active|paused|completed
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    state_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    client: Mapped[Client] = relationship(back_populates="campaigns")


# -----------------------------------------------------------------------------
# Integrations
# -----------------------------------------------------------------------------


class IntegrationConnection(Base):
    """OAuth connection from a tenant to a third-party provider (Google, GHL).

    Tokens stored encrypted with per-tenant key. NEVER read these fields
    without going through the IntegrationToken helpers.
    """

    __tablename__ = "integration_connections"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "client_id",
            "provider",
            "external_account_id",
            name="uq_integration_per_account",
        ),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    # Nullable: tenant-level connections (e.g. Google Workspace) have no client scope.
    # Client-scoped connections (e.g. GHL sub-account per agency client) set this.
    client_id: Mapped[str | None] = mapped_column(
        ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(32), index=True)  # google|ghl|stripe|...
    external_account_id: Mapped[str] = mapped_column(String(128))
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    scopes: Mapped[str] = mapped_column(Text, default="")
    access_token_encrypted: Mapped[str] = mapped_column(Text)
    refresh_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|revoked|error
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    tenant: Mapped[Tenant] = relationship(back_populates="integrations")


# -----------------------------------------------------------------------------
# Jobs (agent invocations) and approvals
# -----------------------------------------------------------------------------


class Job(Base):
    """One agent invocation = one job. Maps to a LangGraph thread_id.

    Approval-pending jobs sit here with status='awaiting_approval' until the
    user clicks Approve/Reject in Slack, at which point a worker resumes
    the LangGraph thread.
    """

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_tenant_client", "tenant_id", "client_id"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str | None] = mapped_column(
        ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True
    )
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    channel_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_message_ts: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), default="queued"
    )  # queued|running|awaiting_approval|resuming|done|failed|rejected
    plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_urls: Mapped[list[str]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class AuditEvent(Base):
    """Append-only log of every meaningful action (tool calls, approvals, errors)."""

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_tenant_ts", "tenant_id", "ts"),
        Index("ix_audit_tenant_client_ts", "tenant_id", "client_id", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str | None] = mapped_column(
        ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64))  # tool_call|approval|error|llm_call
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    args_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_args: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result_status: Mapped[str] = mapped_column(String(32), default="ok")
    model_used: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


# -----------------------------------------------------------------------------
# Living Artifact + Skills (reflexive cognition layer)
# -----------------------------------------------------------------------------


class WorkspaceArtifact(Base):
    """Durable per-thread workspace state distilled after each completed job.

    One row per (tenant_id, client_id, thread_id). The `body` JSONB holds
    key/value facts the agent has learned: client preferences, pipeline names,
    last actions taken, etc. Injected into the agent system prompt every turn.
    """

    __tablename__ = "workspace_artifacts"
    __table_args__ = (
        UniqueConstraint("tenant_id", "client_id", "thread_id", name="uq_artifact_per_thread"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str | None] = mapped_column(
        ForeignKey("clients.id", ondelete="CASCADE"), nullable=True, index=True
    )
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Skill(Base):
    """A promoted tool-sequence shortcut for a (tenant, client) pair.

    The Skill Crystallizer mines audit_events and creates a Skill row when it
    sees the same (tool_name, arg_schema_shape) sequence appear >= 5 times.
    The agent sees skill slugs in its system prompt as workflow shortcuts.
    """

    __tablename__ = "skills"
    __table_args__ = (
        UniqueConstraint("tenant_id", "client_id", "slug", name="uq_skill_per_client"),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str | None] = mapped_column(
        ForeignKey("clients.id", ondelete="CASCADE"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # [{tool_name: str, arg_schema_shape: [[key, type_name]]}]
    tool_sequence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    usage_count: Mapped[int] = mapped_column(default=0)
    promoted_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


# -----------------------------------------------------------------------------
# LLM provider configuration (super-admin runtime switching)
# -----------------------------------------------------------------------------
#
# Two tables back the multi-model switch panel:
#
#   llm_providers          -- one row per *configured* provider (qwen, deepseek,
#                              openai, ...). Holds the encrypted API key plus
#                              optional endpoint override and provider-specific
#                              extra config (e.g. azure deployment_id). Rows
#                              only exist for providers the operator has
#                              actually set up; the static catalog in
#                              packages/lyra_core/llm/catalog.py describes
#                              every provider we *could* use.
#
#   llm_model_assignments  -- one row per tier (`primary`, `cheap`, ...) saying
#                              which provider+model is currently active. The
#                              router reads this on every chat() call (with a
#                              short cache) so flipping a model in the admin UI
#                              propagates within ~30s without a redeploy.


class LlmProvider(Base):
    """Per-vendor credentials configured by the platform super-admin.

    `provider_key` matches a key in `lyra_core.llm.catalog.PROVIDERS`. The
    catalog supplies the LiteLLM model id format and default endpoint; this
    row supplies the secrets and any per-deployment overrides.

    `api_key_encrypted` is encrypted via `common.crypto.encrypt_platform`
    (not per-tenant -- these are platform-level secrets). Never read this
    column directly outside `lyra_core.llm.router`.
    """

    __tablename__ = "llm_providers"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    provider_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_base: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_test_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_test_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    updated_by_email: Mapped[str | None] = mapped_column(String(256), nullable=True)


class LlmModelAssignment(Base):
    """Which provider+model is currently active for each tier.

    Singleton-per-tier (unique on `tier`). Tiers used today: `primary`,
    `cheap`, `embedding`. New tiers are just new rows -- no schema change.
    """

    __tablename__ = "llm_model_assignments"
    __table_args__ = (UniqueConstraint("tier", name="uq_llm_assignment_tier"),)

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tier: Mapped[str] = mapped_column(String(32), index=True)
    provider_key: Mapped[str] = mapped_column(String(32), index=True)
    model_id: Mapped[str] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    updated_by_email: Mapped[str | None] = mapped_column(String(256), nullable=True)
