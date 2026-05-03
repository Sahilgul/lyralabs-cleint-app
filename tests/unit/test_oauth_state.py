"""apps.api.oauth._state."""

from __future__ import annotations

import time

import jwt
import pytest
from lyra_core.common.config import get_settings

from apps.api.oauth._state import _STATE_TTL_SECONDS, decode_state, encode_state


def test_round_trip_with_redirect() -> None:
    token = encode_state("tenant-X", "/integrations")
    tid, rt = decode_state(token)
    assert tid == "tenant-X"
    assert rt == "/integrations"


def test_round_trip_without_redirect() -> None:
    token = encode_state("tenant-X")
    tid, rt = decode_state(token)
    assert tid == "tenant-X"
    assert rt is None


def test_token_includes_correct_claims() -> None:
    token = encode_state("t-1")
    settings = get_settings()
    claims = jwt.decode(
        token, settings.admin_jwt_secret, algorithms=["HS256"], issuer=settings.admin_jwt_issuer
    )
    assert claims["tid"] == "t-1"
    assert claims["iss"] == settings.admin_jwt_issuer
    assert claims["exp"] > time.time()


def test_decode_rejects_bad_signature() -> None:
    settings = get_settings()
    bogus = jwt.encode(
        {"tid": "t", "exp": int(time.time()) + 60, "iss": settings.admin_jwt_issuer},
        "wrong-secret-padded-to-32-bytes!!",
        algorithm="HS256",
    )
    with pytest.raises(jwt.InvalidSignatureError):
        decode_state(bogus)


def test_decode_rejects_expired() -> None:
    settings = get_settings()
    expired = jwt.encode(
        {"tid": "t", "exp": int(time.time()) - 1, "iss": settings.admin_jwt_issuer},
        settings.admin_jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_state(expired)


def test_decode_rejects_wrong_issuer() -> None:
    settings = get_settings()
    bad = jwt.encode(
        {"tid": "t", "exp": int(time.time()) + 60, "iss": "evil"},
        settings.admin_jwt_secret,
        algorithm="HS256",
    )
    with pytest.raises(jwt.InvalidIssuerError):
        decode_state(bad)


def test_state_ttl_constant_is_10_min() -> None:
    assert _STATE_TTL_SECONDS == 600
