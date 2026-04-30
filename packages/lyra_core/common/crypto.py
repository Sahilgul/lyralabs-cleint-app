"""Per-tenant encryption.

Master key (Fernet 32-byte b64) lives in Secret Manager.
Per-tenant key derived via HKDF(master_key, salt=tenant_id) so a leaked
tenant key cannot decrypt other tenants and so we can rotate the master
key by re-encrypting on read.
"""

from __future__ import annotations

import base64
from functools import lru_cache

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .config import get_settings


def _derive_tenant_key(master_key: bytes, tenant_id: str) -> bytes:
    """Derive a 32-byte Fernet key from the master key + tenant id."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=tenant_id.encode("utf-8"),
        info=b"coworker-tenant-token-key",
    )
    raw = hkdf.derive(master_key)
    return base64.urlsafe_b64encode(raw)


@lru_cache(maxsize=512)
def fernet_for_tenant(tenant_id: str) -> Fernet:
    """Return a Fernet keyed for this tenant. Cached for hot path."""
    settings = get_settings()
    master = base64.urlsafe_b64decode(settings.master_encryption_key.encode("utf-8"))
    return Fernet(_derive_tenant_key(master, tenant_id))


def encrypt_for_tenant(tenant_id: str, plaintext: str) -> str:
    return fernet_for_tenant(tenant_id).encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_for_tenant(tenant_id: str, ciphertext: str) -> str:
    return fernet_for_tenant(tenant_id).decrypt(ciphertext.encode("utf-8")).decode("utf-8")


def reencrypt_with_rotation(tenant_id: str, ciphertext: str, old_master: bytes) -> str:
    """Migration helper: decrypt with old key, re-encrypt with current key."""
    old_fernet = Fernet(_derive_tenant_key(old_master, tenant_id))
    new_fernet = fernet_for_tenant(tenant_id)
    rotated = MultiFernet([new_fernet, old_fernet])
    plaintext = rotated.decrypt(ciphertext.encode("utf-8"))
    return new_fernet.encrypt(plaintext).decode("utf-8")
