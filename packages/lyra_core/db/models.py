"""SQLAlchemy 2.0 ORM models.

Tenant = Slack workspace (or Teams tenant). All other tables FK to it.
OAuth tokens are stored encrypted (see common/crypto.py).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

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
    type_annotation_map = {dict[str, Any]: JSONB}


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
    plan: Mapped[str] = mapped_column(String(32), default="trial")  # trial|team|enterprise|cancelled
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
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    external_user_id: Mapped[str] = mapped_column(String(128))
    channel: Mapped[str] = mapped_column(String(16))  # slack|teams|admin
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    email: Mapped[str | None] = mapped_column(String(256), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="member")  # member|admin|owner
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    tenant: Mapped[Tenant] = relationship(back_populates="users")


# -----------------------------------------------------------------------------
# Channel installations
# -----------------------------------------------------------------------------


class SlackInstallation(Base):
    """Persistent Slack OAuth install (used by slack_bolt InstallationStore).

    We mirror the fields slack_bolt expects so we can implement a custom
    InstallationStore backed by Postgres.
    """

    __tablename__ = "slack_installations"
    __table_args__ = (
        Index("ix_slack_install_team_enterprise", "team_id", "enterprise_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
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
            "tenant_id", "provider", "external_account_id", name="uq_integration_per_account"
        ),
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
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

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    channel_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_message_ts: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), default="queued"
    )  # queued|running|awaiting_approval|done|failed|rejected
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
    __table_args__ = (Index("ix_audit_tenant_ts", "tenant_id", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
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
