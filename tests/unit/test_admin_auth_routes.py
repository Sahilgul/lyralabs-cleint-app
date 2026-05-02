"""apps.api.admin.auth_routes — comprehensive test suite.

Covers:
  - Unit tests:      individual function behaviour, edge cases, input validation
  - Behavior tests:  full HTTP request/response flows through the FastAPI app
  - Regression tests: specific bugs that were fixed (duplicate email, timing attack
                       safe 401, password stored hashed not plaintext)
  - Load tests:      concurrent login/register under asyncio.gather to verify
                     no race conditions, no shared-state corruption
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from apps.api.admin.auth_routes import _mint_jwt, _TOKEN_TTL_HOURS, router as auth_router
from lyra_core.common.config import get_settings
from lyra_core.db.models import AdminUser, Client, Tenant


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router, prefix="/admin/auth")
    return app


def _make_tenant(team_id: str = "T123") -> Tenant:
    t = Tenant(external_team_id=team_id, channel="slack", name="Acme")
    t.id = "tenant-uuid-1"
    return t


def _make_user(email: str = "admin@x.com", password: str = "password123") -> AdminUser:
    u = AdminUser()
    u.id = "user-uuid-1"
    u.tenant_id = "tenant-uuid-1"
    u.email = email
    u.password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    u.role = "owner"
    return u


def _decode(token: str) -> dict:
    s = get_settings()
    return jwt.decode(token, s.admin_jwt_secret, algorithms=["HS256"], issuer=s.admin_jwt_issuer)


def _mock_session_cm(execute_side_effect=None, execute_return=None):
    """Build a fake async_session() context manager."""
    s = MagicMock()
    if execute_side_effect:
        s.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        s.execute = AsyncMock(return_value=execute_return)
    s.add = MagicMock()
    s.flush = AsyncMock()
    s.commit = AsyncMock()
    s.refresh = AsyncMock()
    s.__aenter__ = AsyncMock(return_value=s)
    s.__aexit__ = AsyncMock(return_value=False)
    return s


def _scalar(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    r.scalar_one.return_value = value
    return r


def _mock_register_sessions(existing_user=None, tenant_for_update=None):
    """Return two mock sessions for the register endpoint's two DB round-trips."""
    tenant = tenant_for_update or _make_tenant()
    # Session 1: check dupe email → None (no dup); add tenant, flush, add user, commit
    s1 = _mock_session_cm(execute_return=_scalar(existing_user))
    # Session 2: fetch tenant to update external_team_id
    s2 = _mock_session_cm(execute_return=_scalar(tenant))
    return s1, s2


# ===========================================================================
# UNIT TESTS — _mint_jwt()
# ===========================================================================

class TestMintJwt:
    def test_contains_all_required_claims(self):
        token = _mint_jwt("t-1", "a@b.com", "owner")
        claims = _decode(token)
        assert claims["tenant_id"] == "t-1"
        assert claims["email"] == "a@b.com"
        assert claims["role"] == "owner"
        assert claims["iss"] == get_settings().admin_jwt_issuer
        assert "exp" in claims
        assert "iat" in claims

    def test_expires_in_exactly_30_days(self):
        token = _mint_jwt("t-1", "a@b.com", "owner")
        claims = _decode(token)
        assert claims["exp"] - claims["iat"] == _TOKEN_TTL_HOURS * 3600

    def test_token_is_not_yet_expired(self):
        token = _mint_jwt("t-1", "a@b.com", "owner")
        claims = _decode(token)
        assert claims["exp"] > int(time.time())

    def test_different_tenants_produce_different_tokens(self):
        t1 = _mint_jwt("tenant-A", "a@b.com", "owner")
        t2 = _mint_jwt("tenant-B", "a@b.com", "owner")
        assert t1 != t2

    def test_roles_are_embedded(self):
        for role in ("owner", "super_admin", "viewer"):
            claims = _decode(_mint_jwt("t-1", "a@b.com", role))
            assert claims["role"] == role


# ===========================================================================
# UNIT TESTS — password hashing (regression: plaintext must never be stored)
# ===========================================================================

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        pw = "supersecret"
        h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        assert pw not in h

    def test_correct_password_verifies(self):
        pw = "supersecret"
        h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        assert bcrypt.checkpw(pw.encode(), h.encode())

    def test_wrong_password_fails(self):
        h = bcrypt.hashpw(b"correct", bcrypt.gensalt()).decode()
        assert not bcrypt.checkpw(b"wrong", h.encode())

    def test_two_hashes_of_same_password_differ(self):
        """Bcrypt uses random salt — same password must not produce identical hash."""
        pw = b"samepassword"
        h1 = bcrypt.hashpw(pw, bcrypt.gensalt()).decode()
        h2 = bcrypt.hashpw(pw, bcrypt.gensalt()).decode()
        assert h1 != h2


# ===========================================================================
# BEHAVIOR TESTS — POST /admin/auth/register
# ===========================================================================

class TestRegister:
    @pytest.mark.asyncio
    async def test_success_returns_201_with_jwt(self):
        s1, s2 = _mock_register_sessions()

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "admin@x.com", "password": "securepass",
                    "passcode": "7172",
                })

        assert r.status_code == 201
        data = r.json()
        assert data["token_type"] == "bearer"
        claims = _decode(data["access_token"])
        assert claims["email"] == "admin@x.com"
        assert claims["role"] == "owner"
        assert "tenant_id" in claims

    @pytest.mark.asyncio
    async def test_tenant_auto_created_no_team_id_required(self):
        """Registration must succeed without any Slack team_id."""
        s1, s2 = _mock_register_sessions()

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "new@example.com", "password": "securepass", "passcode": "7172",
                })

        assert r.status_code == 201

    @pytest.mark.asyncio
    async def test_email_stored_lowercase(self):
        """Regression: mixed-case email in request must be stored/returned lowercase."""
        stored: list = []
        s1, s2 = _mock_register_sessions()

        original_add = s1.add
        def capture_add(obj):
            stored.append(obj)
        s1.add = capture_add

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "ADMIN@X.COM", "password": "securepass", "passcode": "7172",
                })

        assert r.status_code == 201
        claims = _decode(r.json()["access_token"])
        assert claims["email"] == "admin@x.com"
        users = [obj for obj in stored if isinstance(obj, AdminUser)]
        assert users, "AdminUser was never passed to session.add()"
        assert users[0].email == "admin@x.com"

    @pytest.mark.asyncio
    async def test_wrong_passcode_returns_403(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.post("/admin/auth/register", json={
                "email": "a@x.com", "password": "securepass", "passcode": "0000",
            })
        assert r.status_code == 403
        assert "passcode" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_duplicate_email_returns_409(self):
        existing = _make_user()
        s1 = _mock_session_cm(execute_return=_scalar(existing))

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "admin@x.com", "password": "securepass", "passcode": "7172",
                })
        assert r.status_code == 409
        assert "already registered" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_short_password_returns_422(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.post("/admin/auth/register", json={
                "email": "a@x.com", "password": "short", "passcode": "7172",
            })
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_email_format_returns_422(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.post("/admin/auth/register", json={
                "email": "not-an-email", "password": "securepass", "passcode": "7172",
            })
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_fields_returns_422(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.post("/admin/auth/register", json={"email": "a@x.com"})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_password_is_not_stored_in_plaintext(self):
        """Regression: password must be bcrypt-hashed, never stored as plaintext."""
        stored: list = []
        s1, s2 = _mock_register_sessions()
        s1.add = lambda obj: stored.append(obj)

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                await c.post("/admin/auth/register", json={
                    "email": "a@x.com", "password": "mypassword123", "passcode": "7172",
                })

        users = [obj for obj in stored if isinstance(obj, AdminUser)]
        assert users, "AdminUser was never passed to session.add()"
        assert "mypassword123" not in users[0].password_hash
        assert bcrypt.checkpw(b"mypassword123", users[0].password_hash.encode())


# ===========================================================================
# BEHAVIOR TESTS — POST /admin/auth/login
# ===========================================================================

class TestLogin:
    @pytest.mark.asyncio
    async def test_success_returns_200_with_jwt(self):
        user = _make_user(email="admin@x.com", password="mypassword")
        s = _mock_session_cm(execute_return=_scalar(user))

        with patch("apps.api.admin.auth_routes.async_session", return_value=s):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/login", json={
                    "email": "admin@x.com", "password": "mypassword",
                })

        assert r.status_code == 200
        claims = _decode(r.json()["access_token"])
        assert claims["tenant_id"] == user.tenant_id
        assert claims["email"] == user.email

    @pytest.mark.asyncio
    async def test_wrong_password_returns_401(self):
        user = _make_user(password="correctpassword")
        s = _mock_session_cm(execute_return=_scalar(user))

        with patch("apps.api.admin.auth_routes.async_session", return_value=s):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/login", json={
                    "email": "admin@x.com", "password": "wrongpassword",
                })
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_email_returns_401(self):
        s = _mock_session_cm(execute_return=_scalar(None))

        with patch("apps.api.admin.auth_routes.async_session", return_value=s):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/login", json={
                    "email": "ghost@x.com", "password": "anything",
                })
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_email_lookup_is_case_insensitive(self):
        """Regression: ADMIN@X.COM should authenticate same as admin@x.com."""
        user = _make_user(email="admin@x.com", password="mypassword")
        s = _mock_session_cm(execute_return=_scalar(user))

        with patch("apps.api.admin.auth_routes.async_session", return_value=s):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/login", json={
                    "email": "ADMIN@X.COM", "password": "mypassword",
                })
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_wrong_and_unknown_return_same_error_message(self):
        """Regression: timing/info-leak — both wrong-password and unknown-email
        must return identical error detail so attackers can't enumerate accounts."""
        user = _make_user(password="correct")
        found = _mock_session_cm(execute_return=_scalar(user))
        not_found = _mock_session_cm(execute_return=_scalar(None))

        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            with patch("apps.api.admin.auth_routes.async_session", return_value=found):
                r1 = await c.post("/admin/auth/login", json={"email": "a@x.com", "password": "wrong"})
            with patch("apps.api.admin.auth_routes.async_session", return_value=not_found):
                r2 = await c.post("/admin/auth/login", json={"email": "ghost@x.com", "password": "wrong"})

        assert r1.status_code == r2.status_code == 401
        assert r1.json()["detail"] == r2.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_password_returns_422(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.post("/admin/auth/login", json={"email": "a@x.com"})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_returns_422(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.post("/admin/auth/login", json={})
        assert r.status_code == 422


# ===========================================================================
# REGRESSION TESTS
# ===========================================================================

class TestRegressions:
    @pytest.mark.asyncio
    async def test_register_does_not_reveal_account_info_on_409(self):
        """The 409 detail says 'already registered' but must NOT expose the
        existing user's tenant or any other account info."""
        existing = _make_user()
        s1 = _mock_session_cm(execute_return=_scalar(existing))

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "admin@x.com", "password": "pass1234", "passcode": "7172",
                })
        assert r.status_code == 409
        body = r.json()["detail"]
        assert "tenant" not in body
        assert "uuid" not in body

    @pytest.mark.asyncio
    async def test_jwt_issued_by_login_is_accepted_by_current_admin(self):
        """End-to-end: a token produced by login must pass the existing JWT
        middleware (current_admin dependency) without error."""
        from apps.api.admin.auth import current_admin

        user = _make_user(email="owner@x.com", password="pass1234")
        s = _mock_session_cm(execute_return=_scalar(user))

        with patch("apps.api.admin.auth_routes.async_session", return_value=s):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/login", json={
                    "email": "owner@x.com", "password": "pass1234",
                })

        token = r.json()["access_token"]
        principal = await current_admin(authorization=f"Bearer {token}")
        assert principal.tenant_id == user.tenant_id
        assert principal.email == user.email
        assert principal.role == user.role

    def test_passcode_default_is_7172(self):
        """Regression: the default passcode must stay 7172 until explicitly changed."""
        assert get_settings().admin_register_passcode == "7172"

    @pytest.mark.asyncio
    async def test_pending_tenant_external_id_set_on_register(self):
        """Tenant created during registration must have external_team_id starting with 'pending-'."""
        created_tenants: list[Tenant] = []
        s1, s2 = _mock_register_sessions()

        original_add = s1.add
        def capture_add(obj):
            if isinstance(obj, Tenant):
                created_tenants.append(obj)
        s1.add = capture_add

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "sahil@agency.com", "password": "securepass", "passcode": "7172",
                })

        assert r.status_code == 201
        assert created_tenants, "No Tenant was created"
        assert created_tenants[0].external_team_id.startswith("pending-")


# ===========================================================================
# LOAD TESTS — concurrent requests, no race conditions, no shared state
# ===========================================================================

class TestConcurrentLoad:
    @pytest.mark.asyncio
    async def test_50_concurrent_logins_all_succeed(self):
        """50 simultaneous login requests must all get independent 200 responses."""
        user = _make_user(email="admin@x.com", password="loadtest123")

        async def single_login():
            s = _mock_session_cm(execute_return=_scalar(user))
            with patch("apps.api.admin.auth_routes.async_session", return_value=s):
                async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                    return await c.post("/admin/auth/login", json={
                        "email": "admin@x.com", "password": "loadtest123",
                    })

        results = await asyncio.gather(*[single_login() for _ in range(50)])
        statuses = [r.status_code for r in results]
        assert all(s == 200 for s in statuses), f"Some failed: {statuses}"

    @pytest.mark.asyncio
    async def test_50_concurrent_logins_all_return_valid_jwts(self):
        """Every concurrent login must return a decodable, well-formed JWT."""
        user = _make_user(email="admin@x.com", password="loadtest123")

        async def single_login():
            s = _mock_session_cm(execute_return=_scalar(user))
            with patch("apps.api.admin.auth_routes.async_session", return_value=s):
                async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                    r = await c.post("/admin/auth/login", json={
                        "email": "admin@x.com", "password": "loadtest123",
                    })
                    return r.json()["access_token"]

        tokens = await asyncio.gather(*[single_login() for _ in range(50)])
        for token in tokens:
            claims = _decode(token)
            assert claims["tenant_id"] == user.tenant_id

    @pytest.mark.asyncio
    async def test_concurrent_wrong_password_all_return_401(self):
        """50 concurrent failed logins must all independently return 401."""
        user = _make_user(password="correct")

        async def bad_login():
            s = _mock_session_cm(execute_return=_scalar(user))
            with patch("apps.api.admin.auth_routes.async_session", return_value=s):
                async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                    return await c.post("/admin/auth/login", json={
                        "email": "admin@x.com", "password": "wrong",
                    })

        results = await asyncio.gather(*[bad_login() for _ in range(50)])
        assert all(r.status_code == 401 for r in results)

    @pytest.mark.asyncio
    async def test_mixed_concurrent_login_and_register_no_cross_contamination(self):
        """Login and register running simultaneously must not interfere with
        each other's session mocks or response state."""
        user = _make_user(email="login@x.com", password="loginpass")

        async def do_login():
            s = _mock_session_cm(execute_return=_scalar(user))
            with patch("apps.api.admin.auth_routes.async_session", return_value=s):
                async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                    r = await c.post("/admin/auth/login", json={
                        "email": "login@x.com", "password": "loginpass",
                    })
            return ("login", r.status_code)

        async def do_register():
            s1, s2 = _mock_register_sessions()
            with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
                async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                    r = await c.post("/admin/auth/register", json={
                        "email": "new@x.com", "password": "newpassword", "passcode": "7172",
                    })
            return ("register", r.status_code)

        tasks = [do_login() for _ in range(25)] + [do_register() for _ in range(25)]
        results = await asyncio.gather(*tasks)

        login_results = [s for kind, s in results if kind == "login"]
        register_results = [s for kind, s in results if kind == "register"]

        assert all(s == 200 for s in login_results), f"Login failures: {login_results}"
        assert all(s == 201 for s in register_results), f"Register failures: {register_results}"

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_register_all_rejected_except_first(self):
        """Simulates 10 concurrent requests trying to register the same email.
        The mock returns existing=None for the first, existing=user for the rest,
        matching what the DB would do after the first write commits."""
        existing = _make_user()

        async def do_register(is_first: bool):
            if is_first:
                s1, s2 = _mock_register_sessions()
                with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
                    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                        r = await c.post("/admin/auth/register", json={
                            "email": "admin@x.com", "password": "securepass", "passcode": "7172",
                        })
            else:
                s1 = _mock_session_cm(execute_return=_scalar(existing))
                with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1]):
                    async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                        r = await c.post("/admin/auth/register", json={
                            "email": "admin@x.com", "password": "securepass", "passcode": "7172",
                        })
            return r.status_code

        tasks = [do_register(i == 0) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert results[0] == 201, "First registration should succeed"
        assert all(s == 409 for s in results[1:]), f"Subsequent should be 409: {results[1:]}"


# ===========================================================================
# BEHAVIOR TESTS — GET /admin/auth/slack-install-url
# ===========================================================================

class TestSlackInstallUrl:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.get("/admin/auth/slack-install-url")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_signed_url_for_authenticated_tenant(self):
        from apps.api.oauth._state import decode_state

        token = _mint_jwt("tenant-xyz", "owner@x.com", "owner")
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.get("/admin/auth/slack-install-url", headers={"Authorization": f"Bearer {token}"})

        assert r.status_code == 200
        url = r.json()["url"]
        assert "/oauth/slack/install" in url
        assert "sig=" in url

        # The sig must be a verifiable JWT containing the correct tenant_id.
        sig = url.split("sig=")[1]
        tenant_id, _ = decode_state(sig)
        assert tenant_id == "tenant-xyz"

    @pytest.mark.asyncio
    async def test_sig_expires_within_state_ttl(self):
        import time as time_mod
        from apps.api.oauth._state import decode_state

        token = _mint_jwt("t-ttl", "a@x.com", "owner")
        async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
            r = await c.get("/admin/auth/slack-install-url", headers={"Authorization": f"Bearer {token}"})

        import time as time_mod
        sig = r.json()["url"].split("sig=")[1]
        settings = get_settings()
        claims = jwt.decode(sig, settings.admin_jwt_secret, algorithms=["HS256"], issuer=settings.admin_jwt_issuer)
        remaining = claims["exp"] - int(time_mod.time())
        assert 0 < remaining <= 600  # encode_state uses 600-second TTL


# ===========================================================================
# BEHAVIOR TESTS — registration creates agency_internal Client
# ===========================================================================

class TestRegisterCreatesClient:
    @pytest.mark.asyncio
    async def test_agency_internal_client_created_with_tenant(self):
        """Registration must create a Client(slug='agency_internal') in the same transaction."""
        added_objects: list = []
        s1, s2 = _mock_register_sessions()
        s1.add = lambda obj: added_objects.append(obj)

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                r = await c.post("/admin/auth/register", json={
                    "email": "agency@example.com", "password": "securepass", "passcode": "7172",
                })

        assert r.status_code == 201
        clients = [obj for obj in added_objects if isinstance(obj, Client)]
        assert clients, "No Client was created during registration"
        assert clients[0].slug == "agency_internal"
        assert clients[0].status == "active"

    @pytest.mark.asyncio
    async def test_client_name_matches_email_username(self):
        """The synthetic client name should be the email username part."""
        added_objects: list = []
        s1, s2 = _mock_register_sessions()
        s1.add = lambda obj: added_objects.append(obj)

        with patch("apps.api.admin.auth_routes.async_session", side_effect=[s1, s2]):
            async with AsyncClient(transport=ASGITransport(app=_app()), base_url="http://test") as c:
                await c.post("/admin/auth/register", json={
                    "email": "tehreem@agency.com", "password": "securepass", "passcode": "7172",
                })

        clients = [obj for obj in added_objects if isinstance(obj, Client)]
        assert clients[0].name == "tehreem"
