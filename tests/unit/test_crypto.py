"""lyra_core.common.crypto."""

from __future__ import annotations

import base64

import pytest
from cryptography.fernet import InvalidToken
from lyra_core.common.crypto import (
    _derive_tenant_key,
    decrypt_for_tenant,
    encrypt_for_tenant,
    fernet_for_tenant,
    reencrypt_with_rotation,
)


class TestKeyDerivation:
    def test_returns_url_safe_base64_key(self) -> None:
        master = b"a" * 32
        key = _derive_tenant_key(master, "tenant-x")
        decoded = base64.urlsafe_b64decode(key)
        assert len(decoded) == 32

    def test_different_tenant_yields_different_key(self) -> None:
        master = b"a" * 32
        assert _derive_tenant_key(master, "tenant-a") != _derive_tenant_key(master, "tenant-b")

    def test_same_inputs_yield_same_key(self) -> None:
        master = b"a" * 32
        assert _derive_tenant_key(master, "tenant-x") == _derive_tenant_key(master, "tenant-x")

    def test_different_master_yields_different_key(self) -> None:
        assert _derive_tenant_key(b"a" * 32, "t") != _derive_tenant_key(b"b" * 32, "t")


class TestFernetForTenant:
    def test_caches_per_tenant(self) -> None:
        f1 = fernet_for_tenant("cached-tenant")
        f2 = fernet_for_tenant("cached-tenant")
        assert f1 is f2

    def test_returns_new_instance_per_distinct_tenant(self) -> None:
        f1 = fernet_for_tenant("tenant-x")
        f2 = fernet_for_tenant("tenant-y")
        assert f1 is not f2


class TestRoundTrip:
    @pytest.mark.parametrize(
        "plaintext",
        ["short", "x" * 4096, "with newlines\nand tabs\there", "unicode: é 中文 🎉", ""],
    )
    def test_roundtrip(self, plaintext: str) -> None:
        cipher = encrypt_for_tenant("tenant-rt", plaintext)
        assert decrypt_for_tenant("tenant-rt", cipher) == plaintext

    def test_ciphertext_differs_each_call(self) -> None:
        c1 = encrypt_for_tenant("tenant-rt", "secret")
        c2 = encrypt_for_tenant("tenant-rt", "secret")
        assert c1 != c2


class TestTenantIsolation:
    def test_other_tenant_cannot_decrypt(self) -> None:
        cipher = encrypt_for_tenant("tenant-A", "secret")
        with pytest.raises(InvalidToken):
            decrypt_for_tenant("tenant-B", cipher)

    def test_corrupted_ciphertext_raises(self) -> None:
        cipher = encrypt_for_tenant("tenant-A", "secret")
        with pytest.raises(InvalidToken):
            decrypt_for_tenant("tenant-A", cipher[:-1] + "X")


class TestRotation:
    def test_reencrypt_with_old_master(self) -> None:
        from lyra_core.common.crypto import _derive_tenant_key

        # Simulate ciphertext made with the OLD master
        old_master = b"o" * 32
        from cryptography.fernet import Fernet

        old_fernet = Fernet(_derive_tenant_key(old_master, "tenant-rotate"))
        old_cipher = old_fernet.encrypt(b"secret payload").decode()

        rotated = reencrypt_with_rotation("tenant-rotate", old_cipher, old_master)
        assert decrypt_for_tenant("tenant-rotate", rotated) == "secret payload"
