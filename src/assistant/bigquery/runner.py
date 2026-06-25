"""Read-only BigQuery access with a dry-run cost guard.

Builds on the lean ``BigQueryRunner`` pattern from the assignment brief and adds
the two things a production data agent needs:

1. **A cost guard** — every query is dry-run first to estimate the bytes it will
   scan; if that exceeds the configured ceiling the query is refused *before* it
   runs, and the real job is additionally capped with ``maximum_bytes_billed``
   (belt and suspenders). This is the first line of the resilience/cost story.
2. **Typed results** — :class:`QueryResult` / :class:`QueryEstimate` /
   :class:`ColumnInfo` replace bare DataFrames and dicts, so downstream nodes get
   the row data *and* the observability signals (bytes, duration, job id).

Read-only enforcement at the SQL layer (allow-list, no DML/DDL) lives in
``safety/sql_validator.py`` (Phase 6); this module is the execution boundary.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import google.auth
import pandas as pd
from google.cloud import bigquery

from assistant.config import Settings, get_settings
from assistant.resilience import CircuitBreaker, resilient_call

logger = logging.getLogger(__name__)

# Approximate US on-demand analysis price, used only for human-readable cost
# estimates in logs/traces — never for billing decisions.
_USD_PER_TIB = 6.25
_BYTES_PER_GIB = 1024**3
_BYTES_PER_TIB = 1024**4


class BigQueryError(RuntimeError):
    """Base class for errors raised by :class:`BigQueryRunner`."""


class QueryCostError(BigQueryError):
    """Raised when a query's estimated scan exceeds the configured byte ceiling."""

    def __init__(self, estimate: "QueryEstimate", limit_bytes: int) -> None:
        self.estimate = estimate
        self.limit_bytes = limit_bytes
        super().__init__(
            f"Query would scan {estimate.gib:.2f} GiB (~${estimate.usd:.2f}), "
            f"exceeding the {limit_bytes / _BYTES_PER_GIB:.2f} GiB limit."
        )


@dataclass(frozen=True)
class QueryEstimate:
    """The dry-run cost estimate for a query (no bytes are billed to produce it)."""

    bytes_processed: int

    @property
    def gib(self) -> float:
        return self.bytes_processed / _BYTES_PER_GIB

    @property
    def usd(self) -> float:
        return self.bytes_processed / _BYTES_PER_TIB * _USD_PER_TIB


@dataclass(frozen=True)
class ColumnInfo:
    """A single column's metadata, as returned by :meth:`BigQueryRunner.get_table_schema`."""

    name: str
    type: str
    mode: str
    description: str = ""


@dataclass
class QueryResult:
    """The outcome of a successful query: the rows plus execution telemetry."""

    dataframe: pd.DataFrame
    bytes_processed: int
    bytes_billed: int
    duration_ms: float
    job_id: str | None = None

    @property
    def row_count(self) -> int:
        return len(self.dataframe)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Rows as plain dicts. PII masking (Phase 5) happens downstream of this."""
        return self.dataframe.to_dict(orient="records")


class BigQueryRunner:
    """Executes read-only SQL against the configured dataset, with a cost guard."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str = "bigquery-public-data.thelook_ecommerce",
        max_bytes_billed: int = 2_000_000_000,
        client: bigquery.Client | None = None,
        *,
        max_retries: int = 4,
        retry_base_delay: float = 1.0,
        breaker_threshold: int = 5,
        breaker_cooldown_s: float = 30.0,
    ) -> None:
        self.dataset_id = dataset_id
        self.max_bytes_billed = max_bytes_billed
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._breaker = CircuitBreaker(
            "bigquery", threshold=breaker_threshold, cooldown_s=breaker_cooldown_s
        )
        if client is not None:
            self.client = client  # injected (tests)
        else:
            # Bind the quota project to the credentials so user-based ADC is quiet.
            credentials, _ = google.auth.default(quota_project_id=project_id)
            self.client = bigquery.Client(project=project_id, credentials=credentials)
        logger.info("BigQuery runner ready (dataset=%s)", dataset_id)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "BigQueryRunner":
        """Build a runner from application settings."""
        settings = settings or get_settings()
        return cls(
            project_id=settings.google_cloud_project,
            dataset_id=settings.bq_dataset,
            max_bytes_billed=settings.max_bytes_billed,
            max_retries=settings.llm_max_retries,
            retry_base_delay=settings.llm_retry_base_delay,
            breaker_threshold=settings.circuit_breaker_threshold,
            breaker_cooldown_s=settings.circuit_breaker_cooldown_seconds,
        )

    def _resilient(self, func):
        """Run a BigQuery network call with retry-on-transient + the shared breaker."""
        return resilient_call(
            func,
            breaker=self._breaker,
            max_attempts=self._max_retries,
            base_delay=self._retry_base_delay,
        )

    def dry_run(self, sql: str) -> QueryEstimate:
        """Estimate a query's scan size without running it (validates refs + syntax)."""

        def _job() -> QueryEstimate:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            job = self.client.query(sql, job_config=job_config)
            return QueryEstimate(bytes_processed=int(job.total_bytes_processed or 0))

        return self._resilient(_job)

    def execute_query(self, sql: str, *, max_bytes_billed: int | None = None) -> QueryResult:
        """Dry-run for cost, refuse if over budget, then execute under a hard cap.

        Transient failures (timeouts, 5xx, rate limits) are retried with backoff and
        tracked by a circuit breaker; permanent errors (e.g. SQL syntax) fail fast so
        the graph's self-correction loop can repair them.

        Raises:
            QueryCostError: if the estimated scan exceeds the byte ceiling.
            google.api_core.exceptions.GoogleAPIError: on syntax/execution errors.
        """
        limit = self.max_bytes_billed if max_bytes_billed is None else max_bytes_billed

        estimate = self.dry_run(sql)
        if estimate.bytes_processed > limit:
            raise QueryCostError(estimate, limit)

        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=limit)
        start = time.perf_counter()

        def _run() -> tuple[bigquery.QueryJob, pd.DataFrame]:
            job = self.client.query(sql, job_config=job_config)
            # Analytic report result sets are small, so the REST path is plenty fast;
            # opting out of the BigQuery Storage API keeps our dependency footprint lean.
            return job, job.result().to_dataframe(create_bqstorage_client=False)

        job, dataframe = self._resilient(_run)
        duration_ms = (time.perf_counter() - start) * 1000.0

        result = QueryResult(
            dataframe=dataframe,
            bytes_processed=int(job.total_bytes_processed or 0),
            bytes_billed=int(job.total_bytes_billed or 0),
            duration_ms=duration_ms,
            job_id=job.job_id,
        )
        logger.info(
            "Query ok: %d rows, %.2f GiB scanned, %.0f ms",
            result.row_count,
            estimate.gib,
            duration_ms,
        )
        return result

    def get_table_schema(self, table_name: str) -> list[ColumnInfo]:
        """Return the column metadata for a table in the configured dataset."""
        table = self.client.get_table(f"{self.dataset_id}.{table_name}")
        return [
            ColumnInfo(
                name=field.name,
                type=field.field_type,
                mode=field.mode,
                description=field.description or "",
            )
            for field in table.schema
        ]
