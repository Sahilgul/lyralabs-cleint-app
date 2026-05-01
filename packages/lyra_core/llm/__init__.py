"""Multi-provider LLM configuration.

Three pieces:
  - `catalog`: static, code-resident registry of providers and known models.
  - `router`:  runtime resolver that reads DB-backed credentials + the active
               assignment per tier, with a short in-process cache.
  - admin endpoints (in apps/api/admin/llm.py) call into both to render the
    super-admin model-switcher UI and persist updates.

Adding a new provider later (GLM, Kimi K2, MiniMax, ...) is one entry in
`catalog.PROVIDERS` -- the router and admin endpoints don't need changes.
"""

from .catalog import PROVIDERS, ModelSpec, ProviderSpec
from .router import (
    ResolvedModel,
    invalidate_router_cache,
    list_configured_providers,
    resolve,
    test_provider_connection,
)

__all__ = [
    "PROVIDERS",
    "ModelSpec",
    "ProviderSpec",
    "ResolvedModel",
    "invalidate_router_cache",
    "list_configured_providers",
    "resolve",
    "test_provider_connection",
]
