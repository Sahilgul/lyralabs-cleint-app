"""Centralized settings, validated via pydantic-settings.

All env vars live here. Importing modules NEVER read os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production", "test"] = "development"
    app_base_url: str = "http://localhost:8000"
    admin_base_url: str = "http://localhost:5173"
    log_level: str = "INFO"

    # Postgres
    database_url: str
    database_url_sync: str

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Crypto
    master_encryption_key: str

    # LLM providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    # Alibaba Qwen via DashScope. International endpoint by default; switch to
    # https://dashscope.aliyuncs.com/compatible-mode/v1 for the China region.
    qwen_api_key: str = ""
    qwen_api_base: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    # DeepSeek (deepseek-chat = V3, deepseek-reasoner = R1). Cheap + strong.
    deepseek_api_key: str = ""
    # MiniMax (OpenAI-compatible endpoint at api.minimax.io).
    minimax_api_key: str = ""
    # Kimi / Moonshot AI (OpenAI-compatible endpoint at api.moonshot.ai).
    kimi_api_key: str = ""

    llm_primary_model: str = "deepseek/deepseek-v4-pro"
    llm_critic_model: str = "openai/MiniMax-M2.7"  # final user-facing summary writer
    llm_cheap_model: str = "gemini/gemini-2.5-flash"
    llm_embedding_model: str = "text-embedding-3-small"

    # Three-tier fallback chain — Pro quality (planning / reasoning / critic).
    # Tried in order: Primary → Secondary → Tertiary.
    llm_primary_pro: str = "deepseek/deepseek-v4-pro"
    llm_secondary_pro: str = "openai/MiniMax-M2.7"
    llm_tertiary_pro: str = "openai/kimi-k2.6"

    # Three-tier fallback chain — Flash quality (fast helper calls).
    llm_primary_flash: str = "deepseek/deepseek-v4-flash"
    llm_secondary_flash: str = "openai/MiniMax-M2.5"
    llm_tertiary_flash: str = "openai/kimi-k2.5"

    # Slack
    slack_client_id: str = ""
    slack_client_secret: str = ""
    slack_signing_secret: str = ""
    slack_scopes: str = ""
    slack_user_scopes: str = ""
    slack_install_redirect_url: str = ""
    slack_error_webhook_url: str = ""
    # App-Level Token (xapp-...). Optional. When set, the Socket Mode
    # runner connects via WebSocket and the HTTPS /slack/events endpoint
    # is no longer the primary inbound path. When empty, ARLO falls back
    # to the original Cloud Run HTTPS webhook.
    slack_app_token: str = ""

    # Teams (phase 2)
    teams_app_id: str = ""
    teams_app_password: str = ""

    # Google
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = ""
    google_oauth_scopes: str = ""

    # GHL
    ghl_client_id: str = ""
    ghl_client_secret: str = ""
    ghl_redirect_uri: str = ""
    ghl_scopes: str = ""
    # GHL eval-only credentials (Private Integration Token, no DB / OAuth needed).
    # Used exclusively by tests/eval/test_arlo_ghl_live.py.
    # Get from: GHL sub-account → Settings → Business Info → Private Integrations.
    ghl_eval_token: str = ""
    ghl_eval_location_id: str = ""

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_id_team_monthly: str = ""
    stripe_trial_credit_usd: float = 100.0

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Admin
    admin_jwt_secret: str = ""
    admin_jwt_issuer: str = "lyralabs-admin"
    admin_register_passcode: str = "7172"  # Gate for new admin account creation

    @field_validator("master_encryption_key")
    @classmethod
    def _key_must_be_set(cls, v: str) -> str:
        if not v or v == "replace-me-with-a-fernet-key":
            raise ValueError(
                "MASTER_ENCRYPTION_KEY must be set. "
                "Generate with: python -c 'from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())'"
            )
        return v

    @property
    def slack_scopes_list(self) -> list[str]:
        return [s.strip() for s in self.slack_scopes.split(",") if s.strip()]

    @property
    def google_scopes_list(self) -> list[str]:
        return [s.strip() for s in self.google_oauth_scopes.split(",") if s.strip()]

    @property
    def ghl_scopes_list(self) -> list[str]:
        return [s.strip() for s in self.ghl_scopes.split(" ") if s.strip()]

    @property
    def is_prod(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
