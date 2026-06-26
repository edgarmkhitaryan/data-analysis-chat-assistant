"""Typed application settings, loaded and validated once at startup.

All configuration comes from environment variables (a local ``.env`` in the
prototype; real environment / Secret Manager in production). Call
:func:`get_settings` to obtain a single, cached, validated view of that
configuration. Required values that are missing fail fast at first access with a
clear pydantic error, instead of surfacing as obscure failures deep in a node.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Validated configuration for the assistant. One field per ``.env`` variable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Required credentials ---
    gemini_api_key: SecretStr = Field(alias="GEMINI_API_KEY")
    google_cloud_project: str = Field(alias="GOOGLE_CLOUD_PROJECT")

    # --- BigQuery ---
    bq_dataset: str = Field("bigquery-public-data.thelook_ecommerce", alias="BQ_DATASET")
    max_bytes_billed: int = Field(2_000_000_000, alias="MAX_BYTES_BILLED")
    sql_max_limit: int = Field(1000, alias="SQL_MAX_LIMIT")

    # --- Models ---
    llm_model: str = Field("gemini-3.1-flash-lite", alias="LLM_MODEL")
    llm_model_heavy: str = Field("gemini-3.1-pro-preview", alias="LLM_MODEL_HEAVY")
    embedding_model: str = Field("models/gemini-embedding-001", alias="EMBEDDING_MODEL")

    # --- Agent behavior ---
    max_sql_attempts: int = Field(3, alias="MAX_SQL_ATTEMPTS")
    contextualize_confidence_floor: float = Field(0.6, alias="CONTEXTUALIZE_CONFIDENCE_FLOOR")
    max_sub_questions: int = Field(4, alias="MAX_SUB_QUESTIONS")
    max_history_messages: int = Field(10, alias="MAX_HISTORY_MESSAGES")

    # --- Resilience (retries + circuit breaker for LLM/BigQuery) ---
    llm_max_retries: int = Field(4, alias="LLM_MAX_RETRIES")
    llm_retry_base_delay: float = Field(1.0, alias="LLM_RETRY_BASE_DELAY")
    circuit_breaker_threshold: int = Field(5, alias="CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_cooldown_seconds: float = Field(30.0, alias="CIRCUIT_BREAKER_COOLDOWN_SECONDS")

    # --- Golden Bucket (Hybrid Intelligence) ---
    golden_top_k: int = Field(3, alias="GOLDEN_TOP_K")
    golden_sim_floor: float = Field(0.68, alias="GOLDEN_SIM_FLOOR")
    golden_trios_dir: str = Field("data/golden_trios", alias="GOLDEN_TRIOS_DIR")
    golden_index_dir: str = Field("data/golden_index", alias="GOLDEN_INDEX_DIR")

    # --- Identity & persona ---
    default_persona: str = Field("concise_exec", alias="DEFAULT_PERSONA")
    default_user: str = Field("manager_a", alias="DEFAULT_USER")
    personas_dir: str = Field("data/personas", alias="PERSONAS_DIR")
    app_db_path: str = Field("data/app.db", alias="APP_DB_PATH")
    seed_reports_dir: str = Field("data/seed_reports", alias="SEED_REPORTS_DIR")

    # --- PII masking ---
    # ``NoDecode`` stops pydantic-settings from JSON-decoding the env value so our
    # validator can accept a plain comma-separated list (e.g. "email,postal_code").
    pii_mask_columns: Annotated[list[str], NoDecode] = Field(
        default=["email", "street_address", "postal_code", "latitude", "longitude", "user_geom"],
        alias="PII_MASK_COLUMNS",
    )
    pii_mask_style: Literal["partial", "redact"] = Field("partial", alias="PII_MASK_STYLE")

    # --- Observability ---
    langsmith_api_key: SecretStr | None = Field(None, alias="LANGSMITH_API_KEY")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    traces_dir: str = Field("traces", alias="TRACES_DIR")
    logs_dir: str = Field("logs", alias="LOGS_DIR")

    @field_validator("pii_mask_columns", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept either a comma-separated string (from env) or an actual list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, reading and validating ``.env`` once."""
    return Settings()
