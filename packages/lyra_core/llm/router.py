"""Runtime LLM model router.

Reads the active `(provider, model)` assignment for a tier from Postgres,
joins it with the provider's encrypted credentials + endpoint override,
and returns a `ResolvedModel` that callers pass straight to
`litellm.acompletion`.

Caching strategy
----------------
Hot path: every chat() call goes through `resolve()`. The router keeps a
30-second in-process cache so we don't slam Postgres for every LLM call.
Admin writes call `invalidate_router_cache()` so a model switch
propagates to every worker process within ~30s (worst case: stale
cache served right before the write hits the DB; fixed in <= TTL).

We deliberately don't share cache across processes -- the staleness
window is tiny and the alternative (Redis pub/sub or NOTIFY) buys
seconds at the cost of a lot of moving parts. Revisit if multi-region.

Bootstrap
---------
If no DB row exists for a tier, the router falls back to the
`LLM_PRIMARY_MODEL` / `LLM_CHEAP_MODEL` env vars and reads the matching
provider key from env (`QWEN_API_KEY`, `DEEPSEEK_API_KEY`, etc. via
Settings). This keeps existing deployments working before the operator
touches the admin UI, and gives ops a "break-glass" override (set the
env var, redeploy, no DB write needed).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select

from ..common.config import get_settings
from ..common.crypto import decrypt_platform
from ..common.logging import get_logger
from ..db.models import LlmModelAssignment, LlmProvider
from ..db.session import async_session
from .catalog import PROVIDERS, ModelSpec, ProviderSpec, model_spec, provider_for_model

log = get_logger(__name__)

Tier = Literal["primary", "cheap", "embedding"]

_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class ResolvedModel:
    """Everything `litellm.acompletion` needs to call this model.

    `extra_kwargs` carries provider-specific kwargs (e.g. `organization`
    for OpenAI, `api_version` + `deployment_id` for Azure) that the
    operator filled in via the admin UI's `extra_config` JSON editor.
    """

    tier: str
    provider_key: str
    model_id: str
    api_key: str | None
    api_base: str | None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    source: Literal["db", "env"] = "db"


# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_cache: dict[str, ResolvedModel] = {}
_cache_loaded_at: float = 0.0
_cache_lock = asyncio.Lock()


def invalidate_router_cache() -> None:
    """Drop the cached resolutions; next `resolve()` call hits the DB.

    Call this from any admin endpoint that mutates llm_providers or
    llm_model_assignments. Safe to call from any thread/loop -- it's
    just a couple of dict assignments.
    """
    global _cache_loaded_at
    _cache.clear()
    _cache_loaded_at = 0.0


def _cache_fresh() -> bool:
    return _cache_loaded_at > 0 and (time.monotonic() - _cache_loaded_at) < _CACHE_TTL_SECONDS


async def _load_cache() -> None:
    """Read all (provider, assignment) rows once and populate the cache."""
    global _cache_loaded_at

    async with async_session() as s:
        provider_rows = (await s.execute(select(LlmProvider))).scalars().all()
        assignment_rows = (
            (await s.execute(select(LlmModelAssignment))).scalars().all()
        )

    providers_by_key = {p.provider_key: p for p in provider_rows}

    new_cache: dict[str, ResolvedModel] = {}
    for a in assignment_rows:
        prov = providers_by_key.get(a.provider_key)
        if prov is None or not prov.enabled:
            log.warning(
                "llm.router.skip_assignment",
                tier=a.tier,
                reason="provider_missing_or_disabled",
                provider=a.provider_key,
            )
            continue
        api_key = (
            decrypt_platform(prov.api_key_encrypted) if prov.api_key_encrypted else None
        )
        spec = PROVIDERS.get(a.provider_key)
        api_base = prov.api_base or (spec.default_api_base if spec else None)
        new_cache[a.tier] = ResolvedModel(
            tier=a.tier,
            provider_key=a.provider_key,
            model_id=a.model_id,
            api_key=api_key,
            api_base=api_base,
            extra_kwargs=dict(prov.extra_config or {}),
            source="db",
        )

    _cache.clear()
    _cache.update(new_cache)
    _cache_loaded_at = time.monotonic()


def _resolve_from_env(tier: Tier) -> ResolvedModel:
    """Bootstrap fallback: build a ResolvedModel from env-var settings.

    Used when no `llm_model_assignments` row exists for the tier yet
    (fresh deploy) -- keeps the agent working while the operator hasn't
    touched the admin UI. Also useful as a documented "break-glass":
    set LLM_PRIMARY_MODEL in env, redeploy, no DB write.
    """
    settings = get_settings()
    if tier == "primary":
        model_id = settings.llm_primary_model
    elif tier == "cheap":
        model_id = settings.llm_cheap_model
    else:  # embedding
        model_id = settings.llm_embedding_model

    spec = provider_for_model(model_id)
    api_key = _env_api_key_for(spec) if spec else None
    api_base = spec.default_api_base if spec else None
    return ResolvedModel(
        tier=tier,
        provider_key=spec.key if spec else "unknown",
        model_id=model_id,
        api_key=api_key,
        api_base=api_base,
        extra_kwargs={},
        source="env",
    )


def _env_api_key_for(spec: ProviderSpec) -> str | None:
    """Map a provider catalog entry to the corresponding env-var setting."""
    settings = get_settings()
    return {
        "qwen": settings.qwen_api_key or None,
        "deepseek": settings.deepseek_api_key or None,
        "openai": settings.openai_api_key or None,
        "anthropic": settings.anthropic_api_key or None,
        "gemini": settings.google_api_key or None,
    }.get(spec.key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def resolve(tier: Tier) -> ResolvedModel:
    """Return the active model for `tier`, with credentials + endpoint.

    Cached for `_CACHE_TTL_SECONDS`. Falls back to env-var settings if
    no DB assignment exists. Always returns a value -- never raises for
    "no model configured" because that would silently break the agent.
    Caller is responsible for handling auth errors at LiteLLM call time.
    """
    if not _cache_fresh():
        async with _cache_lock:
            if not _cache_fresh():
                try:
                    await _load_cache()
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "llm.router.cache_load_failed",
                        error=str(exc),
                        action="falling_back_to_env",
                    )

    cached = _cache.get(tier)
    if cached is not None:
        return cached
    return _resolve_from_env(tier)


async def list_configured_providers() -> list[dict[str, Any]]:
    """For admin UI: every catalog entry merged with its DB config (if any).

    Returns one dict per provider in the static catalog, augmented with:
      - configured (bool): is there a DB row?
      - has_api_key (bool): is the key set?
      - api_base, extra_config, enabled, last_test_status (if configured)

    Never returns the API key itself.
    """
    async with async_session() as s:
        rows = (await s.execute(select(LlmProvider))).scalars().all()
    by_key = {r.provider_key: r for r in rows}

    out: list[dict[str, Any]] = []
    for key, spec in PROVIDERS.items():
        row = by_key.get(key)
        out.append(
            {
                "key": key,
                "display_name": spec.display_name,
                "litellm_prefix": spec.litellm_prefix,
                "default_api_base": spec.default_api_base,
                "docs_url": spec.docs_url,
                "extra_config_keys": spec.extra_config_keys,
                "known_models": [
                    {
                        "id": m.id,
                        "display_name": m.display_name,
                        "context_window": m.context_window,
                        "tier_hint": m.tier_hint,
                        "notes": m.notes,
                    }
                    for m in spec.known_models
                ],
                "configured": row is not None,
                "enabled": bool(row.enabled) if row else False,
                "has_api_key": bool(row and row.api_key_encrypted),
                "api_base": row.api_base if row else None,
                "extra_config": dict(row.extra_config or {}) if row else {},
                "last_tested_at": row.last_tested_at.isoformat() if row and row.last_tested_at else None,
                "last_test_status": row.last_test_status if row else None,
                "last_test_error": row.last_test_error if row else None,
                "updated_by_email": row.updated_by_email if row else None,
            }
        )
    return out


async def test_provider_connection(
    provider_key: str, model_id: str, *, timeout_s: float = 15.0
) -> tuple[bool, str | None]:
    """Send a minimal completion request to verify the credentials work.

    Returns (ok, error_message). Updates `last_tested_at` + status on
    the DB row so the admin UI can show a green/red indicator without
    re-pinging.
    """
    from datetime import UTC, datetime

    import litellm

    async with async_session() as s:
        prov = (
            await s.execute(
                select(LlmProvider).where(LlmProvider.provider_key == provider_key)
            )
        ).scalar_one_or_none()
        if prov is None:
            return False, f"provider {provider_key!r} is not configured"

        api_key = decrypt_platform(prov.api_key_encrypted) if prov.api_key_encrypted else None
        spec = PROVIDERS.get(provider_key)
        api_base = prov.api_base or (spec.default_api_base if spec else None)
        extra = dict(prov.extra_config or {})

    ok = True
    err: str | None = None
    try:
        await asyncio.wait_for(
            litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
                temperature=0,
                api_key=api_key,
                api_base=api_base,
                **extra,
            ),
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        ok = False
        err = str(exc)[:500]

    async with async_session() as s:
        prov = (
            await s.execute(
                select(LlmProvider).where(LlmProvider.provider_key == provider_key)
            )
        ).scalar_one_or_none()
        if prov is not None:
            prov.last_tested_at = datetime.now(UTC)
            prov.last_test_status = "ok" if ok else "error"
            prov.last_test_error = err
            await s.commit()

    return ok, err


def known_model_for(model_id: str) -> ModelSpec | None:
    """Re-export from catalog for admin endpoint convenience."""
    return model_spec(model_id)
