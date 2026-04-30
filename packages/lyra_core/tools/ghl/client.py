"""Thin async httpx client for GoHighLevel v2 API.

Handles:
  - Base URL + Version header (required by GHL).
  - Bearer auth from a ProviderCredentials.
  - Retries on 429/5xx with exponential backoff (tenacity).
  - Returns parsed JSON or raises ToolError.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..base import ToolError
from ..credentials import ProviderCredentials

GHL_BASE = "https://services.leadconnectorhq.com"
GHL_API_VERSION = "2021-07-28"

_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class GhlClient:
    def __init__(self, creds: ProviderCredentials) -> None:
        self._creds = creds

    @property
    def location_id(self) -> str | None:
        return self._creds.metadata.get("location_id") or self._creds.external_account_id

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._creds.access_token}",
            "Version": GHL_API_VERSION,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{GHL_BASE}{path}"

        async def _attempt() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
                resp = await c.request(
                    method, url, params=params, json=json, headers=self._headers()
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"retryable {resp.status_code}", request=resp.request, response=resp
                )
            if resp.status_code >= 400:
                raise ToolError(f"GHL {method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json() if resp.content else {}

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            reraise=True,
        ):
            with attempt:
                return await _attempt()
        raise ToolError("unreachable")  # pragma: no cover
