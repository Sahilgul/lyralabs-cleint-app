"""lyra_core.db.models — declarative schema sanity tests.

We avoid spinning a real Postgres here. Instead, inspect the ORM metadata
to verify columns, defaults, indices, and constraints are wired correctly.
"""

from __future__ import annotations

from lyra_core.db.models import (
    AuditEvent,
    Base,
    IntegrationConnection,
    Job,
    SlackInstallation,
    Tenant,
    User,
)


def test_all_tables_registered() -> None:
    expected = {
        "tenants",
        "users",
        "slack_installations",
        "integration_connections",
        "jobs",
        "audit_events",
        "llm_providers",
        "llm_model_assignments",
        "clients",
        "campaigns",
        "workspace_artifacts",
        "skills",
        "admin_users",
    }
    assert set(Base.metadata.tables.keys()) == expected


class TestTenantSchema:
    def test_columns(self) -> None:
        cols = {c.name for c in Tenant.__table__.columns}
        assert {
            "id",
            "external_team_id",
            "channel",
            "name",
            "plan",
            "status",
            "trial_credit_remaining_usd",
            "settings",
            "stripe_customer_id",
            "stripe_subscription_id",
            "created_at",
            "updated_at",
        } <= cols

    def test_external_team_id_is_unique(self) -> None:
        col = Tenant.__table__.c.external_team_id
        assert col.unique

    def test_default_plan_is_trial(self) -> None:
        # default lives on the Column; instantiation alone won't populate it
        # without flushing. Test the column default explicitly.
        assert Tenant.__table__.c.plan.default.arg == "trial"
        assert Tenant.__table__.c.status.default.arg == "active"
        assert Tenant.__table__.c.trial_credit_remaining_usd.default.arg == 100.0


class TestUserSchema:
    def test_unique_constraint_on_tenant_user_channel(self) -> None:
        constraints = {c.name for c in User.__table__.constraints if hasattr(c, "name") and c.name}
        assert "uq_user_per_tenant" in constraints

    def test_default_role_member(self) -> None:
        assert User.__table__.c.role.default.arg == "member"


class TestSlackInstallationSchema:
    def test_token_columns_are_text(self) -> None:
        for name in ("bot_token_encrypted", "bot_refresh_token_encrypted"):
            col = SlackInstallation.__table__.c[name]
            assert col.type.python_type is str

    def test_index_on_team_and_enterprise(self) -> None:
        idx_names = {i.name for i in SlackInstallation.__table__.indexes}
        assert "ix_slack_install_team_enterprise" in idx_names

    def test_is_enterprise_install_default_false(self) -> None:
        assert SlackInstallation.__table__.c.is_enterprise_install.default.arg is False


class TestIntegrationConnectionSchema:
    def test_unique_per_account_constraint(self) -> None:
        constraints = {
            c.name
            for c in IntegrationConnection.__table__.constraints
            if hasattr(c, "name") and c.name
        }
        assert "uq_integration_per_account" in constraints

    def test_metadata_column_renamed(self) -> None:
        # Pythonic name `metadata_` -> SQL column `metadata` (since "metadata"
        # collides with SQLAlchemy's MetaData attr).
        cols = {c.name for c in IntegrationConnection.__table__.columns}
        assert "metadata" in cols
        # Mapped attribute is metadata_:
        assert hasattr(IntegrationConnection, "metadata_")

    def test_status_default_active(self) -> None:
        assert IntegrationConnection.__table__.c.status.default.arg == "active"


class TestJobSchema:
    def test_status_default_queued(self) -> None:
        assert Job.__table__.c.status.default.arg == "queued"

    def test_indices_on_thread_and_tenant(self) -> None:
        indexed = {c.name for c in Job.__table__.columns if c.index}
        assert {"tenant_id", "thread_id"} <= indexed

    def test_artifact_urls_default_empty_list(self) -> None:
        # Callable default -> wrapped in CallableColumnDefault; .arg is the callable.
        # SQLAlchemy wraps it; comparing by name is robust.
        d = Job.__table__.c.artifact_urls.default
        assert d.is_callable
        assert getattr(d.arg, "__name__", "") == "list"
        assert Job.__table__.c.cost_usd.default.arg == 0.0


class TestAuditEventSchema:
    def test_composite_index_tenant_ts(self) -> None:
        idx_names = {i.name for i in AuditEvent.__table__.indexes}
        assert "ix_audit_tenant_ts" in idx_names

    def test_default_status_ok(self) -> None:
        assert AuditEvent.__table__.c.result_status.default.arg == "ok"
