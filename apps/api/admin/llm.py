"""Super-admin REST API for runtime LLM model switching.

Backs the model-switcher panel in `lyralabs-admin-ui`. Three concerns:

  1. Catalog discovery     (`GET /admin/llm/catalog`)
       What providers + models do we know about? Pure static data;
       returned even before any provider is configured. Frontend uses
       this to populate dropdowns.

  2. Provider credentials  (`/admin/llm/providers/...`)
       Set / clear / list per-vendor API keys, optional endpoint
       overrides, and provider-specific extra config. Keys are
       encrypted with the platform fernet key on write and never
       returned by any GET endpoint.

  3. Active assignment     (`/admin/llm/active/...`)
       Which provider+model is live for `primary` / `cheap` /
       `embedding` right now. Writes invalidate the router cache so
       the change shows up on the next chat call (~30s worst case
       for any in-flight cache).

All endpoints require the `super_admin` role. Tenant admins keep
using `/admin/me`, `/admin/jobs` etc. and never see this surface.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from lyra_core.common.crypto import encrypt_platform
from lyra_core.db.models import LlmModelAssignment, LlmProvider
from lyra_core.db.session import get_session
from lyra_core.llm.catalog import PROVIDERS, model_spec
from lyra_core.llm.router import (
    invalidate_router_cache,
    list_configured_providers,
    test_provider_connection,
)
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import CurrentSuperAdmin

router = APIRouter()


# --- DTOs ---------------------------------------------------------------------


class ModelOut(BaseModel):
    id: str
    display_name: str
    context_window: int
    tier_hint: str
    notes: str = ""


class CatalogProviderOut(BaseModel):
    """Provider entry from the static catalog (no DB state, no secrets)."""

    key: str
    display_name: str
    litellm_prefix: str
    default_api_base: str | None
    docs_url: str
    extra_config_keys: list[str]
    known_models: list[ModelOut]


class ConfiguredProviderOut(CatalogProviderOut):
    """Catalog entry + DB state. API key is never serialized."""

    configured: bool
    enabled: bool
    has_api_key: bool
    api_base: str | None
    extra_config: dict[str, Any]
    last_tested_at: str | None
    last_test_status: str | None
    last_test_error: str | None
    updated_by_email: str | None


class ProviderUpsertIn(BaseModel):
    api_key: str | None = Field(
        default=None,
        description=(
            "Plaintext API key. Encrypted on write. Pass null to leave the "
            "stored key unchanged; pass empty string to clear it."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description="Optional endpoint override. Null = use catalog default.",
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific extra kwargs forwarded to litellm.acompletion "
            "(e.g. organization for OpenAI, api_version for Azure)."
        ),
    )
    enabled: bool = True


class AssignmentOut(BaseModel):
    tier: str
    provider_key: str
    model_id: str
    notes: str | None
    updated_at: str
    updated_by_email: str | None


class AssignmentSetIn(BaseModel):
    provider_key: str
    model_id: str
    notes: str | None = None


class TestConnectionIn(BaseModel):
    model_id: str = Field(description="Full LiteLLM model id, e.g. dashscope/qwen-turbo")


class TestConnectionOut(BaseModel):
    ok: bool
    error: str | None


# --- Catalog ------------------------------------------------------------------


@router.get("/catalog", response_model=list[CatalogProviderOut])
async def get_catalog(_: CurrentSuperAdmin) -> list[CatalogProviderOut]:
    """Return every provider/model the code knows about. No DB hit."""
    return [
        CatalogProviderOut(
            key=spec.key,
            display_name=spec.display_name,
            litellm_prefix=spec.litellm_prefix,
            default_api_base=spec.default_api_base,
            docs_url=spec.docs_url,
            extra_config_keys=list(spec.extra_config_keys),
            known_models=[
                ModelOut(
                    id=m.id,
                    display_name=m.display_name,
                    context_window=m.context_window,
                    tier_hint=m.tier_hint,
                    notes=m.notes,
                )
                for m in spec.known_models
            ],
        )
        for spec in PROVIDERS.values()
    ]


# --- Providers ----------------------------------------------------------------


@router.get("/providers", response_model=list[ConfiguredProviderOut])
async def list_providers(_: CurrentSuperAdmin) -> list[ConfiguredProviderOut]:
    """Catalog joined with DB state. Used to render the credential editor."""
    rows = await list_configured_providers()
    return [ConfiguredProviderOut(**row) for row in rows]


@router.put("/providers/{provider_key}", response_model=ConfiguredProviderOut)
async def upsert_provider(
    provider_key: str,
    body: ProviderUpsertIn,
    admin: CurrentSuperAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> ConfiguredProviderOut:
    if provider_key not in PROVIDERS:
        raise HTTPException(404, f"unknown provider {provider_key!r}")

    row = (
        await s.execute(select(LlmProvider).where(LlmProvider.provider_key == provider_key))
    ).scalar_one_or_none()

    if row is None:
        row = LlmProvider(provider_key=provider_key)
        s.add(row)

    if body.api_key is not None:
        row.api_key_encrypted = encrypt_platform(body.api_key) if body.api_key else None
    row.api_base = body.api_base or None
    row.extra_config = dict(body.extra_config or {})
    row.enabled = bool(body.enabled)
    row.updated_by_email = admin.email
    # Clear stale test results so the UI doesn't show a green tick
    # against credentials we just changed.
    row.last_test_status = None
    row.last_test_error = None

    await s.commit()
    invalidate_router_cache()

    rows = await list_configured_providers()
    out = next(r for r in rows if r["key"] == provider_key)
    return ConfiguredProviderOut(**out)


@router.delete("/providers/{provider_key}")
async def delete_provider(
    provider_key: str,
    admin: CurrentSuperAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    row = (
        await s.execute(select(LlmProvider).where(LlmProvider.provider_key == provider_key))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "provider is not configured")

    # Refuse to delete a provider that's currently assigned to a tier --
    # otherwise the next chat call would silently fall back to env or
    # error out. Force the operator to flip the assignment first.
    in_use = (
        (
            await s.execute(
                select(LlmModelAssignment.tier).where(
                    LlmModelAssignment.provider_key == provider_key
                )
            )
        )
        .scalars()
        .all()
    )
    if in_use:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"provider {provider_key!r} is assigned to tiers {sorted(in_use)}; "
            "switch those tiers to a different provider first",
        )

    await s.delete(row)
    await s.commit()
    invalidate_router_cache()
    return {"status": "deleted"}


@router.post("/providers/{provider_key}/test", response_model=TestConnectionOut)
async def test_provider(
    provider_key: str,
    body: TestConnectionIn,
    _: CurrentSuperAdmin,
) -> TestConnectionOut:
    """Send a 4-token ping using the stored credentials. Updates the row."""
    if provider_key not in PROVIDERS:
        raise HTTPException(404, f"unknown provider {provider_key!r}")
    spec = model_spec(body.model_id)
    if spec is None:
        # Allow custom IDs not in the catalog (new model on a known provider)
        # but still warn via the response if it's clearly not for this provider.
        prefix_ok = body.model_id.startswith(PROVIDERS[provider_key].litellm_prefix + "/")
        if not prefix_ok:
            raise HTTPException(
                400,
                f"model {body.model_id!r} doesn't match provider {provider_key!r} prefix",
            )
    ok, err = await test_provider_connection(provider_key, body.model_id)
    return TestConnectionOut(ok=ok, error=err)


# --- Active assignment --------------------------------------------------------


@router.get("/active", response_model=list[AssignmentOut])
async def list_assignments(
    _: CurrentSuperAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> list[AssignmentOut]:
    rows = (await s.execute(select(LlmModelAssignment))).scalars().all()
    return [
        AssignmentOut(
            tier=r.tier,
            provider_key=r.provider_key,
            model_id=r.model_id,
            notes=r.notes,
            updated_at=r.updated_at.isoformat(),
            updated_by_email=r.updated_by_email,
        )
        for r in rows
    ]


@router.put("/active/{tier}", response_model=AssignmentOut)
async def set_assignment(
    tier: str,
    body: AssignmentSetIn,
    admin: CurrentSuperAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> AssignmentOut:
    if tier not in {"primary", "cheap", "embedding"}:
        raise HTTPException(400, f"unknown tier {tier!r}")
    if body.provider_key not in PROVIDERS:
        raise HTTPException(400, f"unknown provider {body.provider_key!r}")

    # Don't let the operator activate a provider they haven't given keys to.
    prov = (
        await s.execute(select(LlmProvider).where(LlmProvider.provider_key == body.provider_key))
    ).scalar_one_or_none()
    if prov is None or not prov.enabled or not prov.api_key_encrypted:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"provider {body.provider_key!r} must be configured + enabled "
            "with an API key before it can be assigned",
        )

    row = (
        await s.execute(select(LlmModelAssignment).where(LlmModelAssignment.tier == tier))
    ).scalar_one_or_none()
    if row is None:
        row = LlmModelAssignment(tier=tier)
        s.add(row)
    row.provider_key = body.provider_key
    row.model_id = body.model_id
    row.notes = body.notes
    row.updated_by_email = admin.email

    await s.commit()
    await s.refresh(row)
    invalidate_router_cache()

    return AssignmentOut(
        tier=row.tier,
        provider_key=row.provider_key,
        model_id=row.model_id,
        notes=row.notes,
        updated_at=row.updated_at.isoformat(),
        updated_by_email=row.updated_by_email,
    )


@router.delete("/active/{tier}")
async def clear_assignment(
    tier: str,
    _: CurrentSuperAdmin,
    s: Annotated[AsyncSession, Depends(get_session)],
) -> dict[str, str]:
    """Clear a tier's assignment -- router falls back to env-var settings."""
    row = (
        await s.execute(select(LlmModelAssignment).where(LlmModelAssignment.tier == tier))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "no assignment for that tier")
    await s.delete(row)
    await s.commit()
    invalidate_router_cache()
    return {"status": "cleared"}
