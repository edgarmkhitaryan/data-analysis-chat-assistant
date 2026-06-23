"""Shared resources injected into the graph's nodes.

Bundling the long-lived resources (the BigQuery runner, settings) in one object
and passing it to nodes keeps construction in a single place and makes nodes
trivially testable: a test can build :class:`AgentDeps` with a fake runner.
"""

from dataclasses import dataclass

from assistant.bigquery import BigQueryRunner
from assistant.config import Settings, get_settings
from assistant.golden import GoldenRetriever


@dataclass(frozen=True)
class AgentDeps:
    """Resources the agent's nodes depend on."""

    runner: BigQueryRunner
    retriever: GoldenRetriever
    settings: Settings

    @classmethod
    def create(cls, settings: Settings | None = None) -> "AgentDeps":
        """Build dependencies from application settings (the production path)."""
        settings = settings or get_settings()
        return cls(
            runner=BigQueryRunner.from_settings(settings),
            retriever=GoldenRetriever.from_settings(settings),
            settings=settings,
        )
