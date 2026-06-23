"""Phase 0 smoke test: verify the prototype's external dependencies are reachable.

Run with ``make check`` (or ``python scripts/check_access.py``). It runs four
independent checks — configuration, Gemini chat, Gemini embeddings, and a real
BigQuery query against the public ``thelook_ecommerce`` dataset — and prints a
clear PASS/FAIL summary. The process exits non-zero if any check fails, so it
doubles as a CI readiness gate.

It deliberately holds no application logic: it exercises the same configuration
and services the agent uses, and nothing more.
"""

import sys
from pathlib import Path

# Make ``src/`` importable when this script is run directly, even before an
# editable install (`pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rich.console import Console  # noqa: E402 — must follow the path bootstrap above
from rich.table import Table  # noqa: E402

from assistant.config import Settings, get_settings  # noqa: E402

console = Console()


def check_config() -> Settings:
    """Load and validate settings, then echo the active targets."""
    settings = get_settings()
    console.print(
        f"  project=[cyan]{settings.google_cloud_project}[/]  "
        f"model=[cyan]{settings.llm_model}[/]  "
        f"dataset=[cyan]{settings.bq_dataset}[/]"
    )
    return settings


def check_gemini_chat(settings: Settings) -> None:
    """Confirm the chat model answers a trivial prompt."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=0,
    )
    reply = llm.invoke("Reply with exactly one word: hello").content
    console.print(f"  {settings.llm_model} replied: [green]{reply!r}[/]")


def check_gemini_embeddings(settings: Settings) -> None:
    """Confirm the embedding model returns a vector (used by the Golden Bucket)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embedder = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
    )
    vector = embedder.embed_query("retail revenue by product category")
    console.print(f"  {settings.embedding_model} returned a [green]{len(vector)}-dim[/] vector")


def check_bigquery(settings: Settings) -> None:
    """Run a tiny, cost-guarded query and print a real row from the dataset."""
    import google.auth
    from google.cloud import bigquery

    # Bind the quota project to the credentials so user-based ADC doesn't warn.
    credentials, _ = google.auth.default(quota_project_id=settings.google_cloud_project)
    client = bigquery.Client(project=settings.google_cloud_project, credentials=credentials)
    sql = f"""
        SELECT id, name, category, retail_price
        FROM `{settings.bq_dataset}.products`
        ORDER BY retail_price DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=settings.max_bytes_billed)
    row = next(iter(client.query(sql, job_config=job_config).result()))
    console.print(
        f"  top product: [green]{row.name}[/] "
        f"(category={row.category}, price=${row.retail_price:.2f})"
    )


def _print_summary(results: list[tuple[str, bool]]) -> None:
    table = Table(title="Summary", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result", justify="center")
    for name, ok in results:
        table.add_row(name, "[green]PASS[/]" if ok else "[red]FAIL[/]")
    console.print()
    console.print(table)


def main() -> int:
    console.rule("[bold]Data Analysis Chat Assistant — access check")
    results: list[tuple[str, bool]] = []

    console.print("\n[bold]Configuration[/]")
    try:
        settings = check_config()
        results.append(("Configuration", True))
    except Exception as exc:  # noqa: BLE001 — surface any failure to the operator
        console.print(f"  [red]FAILED[/]: {exc}")
        _print_summary([("Configuration", False)])
        console.print("\n[bold red]Cannot continue without valid configuration.[/]")
        return 1

    dependent_checks = (
        ("Gemini chat", check_gemini_chat),
        ("Gemini embeddings", check_gemini_embeddings),
        ("BigQuery query", check_bigquery),
    )
    for name, check in dependent_checks:
        console.print(f"\n[bold]{name}[/]")
        try:
            check(settings)
            results.append((name, True))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the operator
            console.print(f"  [red]FAILED[/]: {exc}")
            results.append((name, False))

    _print_summary(results)
    all_ok = all(ok for _, ok in results)
    console.print(
        "\n[bold green]All systems go — environment is ready.[/]"
        if all_ok
        else "\n[bold red]Some checks failed — see the errors above.[/]"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
