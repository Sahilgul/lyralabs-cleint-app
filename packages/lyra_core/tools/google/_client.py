"""Helpers for building google-api-python-client services from credentials."""

from __future__ import annotations

from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from ..credentials import ProviderCredentials


def google_creds(creds: ProviderCredentials) -> Credentials:
    """Build a google.oauth2.Credentials object from our ProviderCredentials."""
    return Credentials(
        token=creds.access_token,
        refresh_token=creds.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",  # noqa: S106 public OAuth token endpoint URL, not a secret
        scopes=creds.scopes.split(" ") if creds.scopes else None,
    )


def drive_service(creds: ProviderCredentials) -> Any:
    return build("drive", "v3", credentials=google_creds(creds), cache_discovery=False)


def docs_service(creds: ProviderCredentials) -> Any:
    return build("docs", "v1", credentials=google_creds(creds), cache_discovery=False)


def sheets_service(creds: ProviderCredentials) -> Any:
    return build("sheets", "v4", credentials=google_creds(creds), cache_discovery=False)


def calendar_service(creds: ProviderCredentials) -> Any:
    return build("calendar", "v3", credentials=google_creds(creds), cache_discovery=False)


def slides_service(creds: ProviderCredentials) -> Any:
    return build("slides", "v1", credentials=google_creds(creds), cache_discovery=False)
