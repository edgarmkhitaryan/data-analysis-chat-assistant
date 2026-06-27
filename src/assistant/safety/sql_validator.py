"""sqlglot-based read-only SQL validator (plan/007 §2).

Replaces the Phase 2 string/regex gate with an AST-based one — robust against
comments, casing, and formatting tricks that fool substring matching. It enforces,
before any query runs:

1. **Single statement** — exactly one statement (no stacked ``; DROP ...``).
2. **Read-only** — the statement must be a ``SELECT`` (or a CTE/set-operation of
   SELECTs); any DML/DDL node anywhere in the tree is rejected.
3. **Table allow-list** — every referenced table must be one of the four
   ``thelook_ecommerce`` tables in the configured dataset (CTE names are exempt).
   Read-only ``INFORMATION_SCHEMA`` metadata views (column/table names + types) are
   also allowed, but only when scoped to that same dataset — so database-structure
   questions can be answered without ever exposing row data or PII.
4. **Mandatory LIMIT** — a sane ``LIMIT`` is injected when absent and clamped when
   it exceeds the cap, protecting the CLI and cost.

The dry-run *cost* guard (bytes-scanned ceiling) lives at the execution boundary
in :class:`assistant.bigquery.runner.BigQueryRunner` (it needs the network); this
module is pure and offline-testable. Validation failures are returned as
structured errors the self-correction loop can act on (plan/008).
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

# The four tables the agent may ever touch (matches the schema provider).
DEFAULT_ALLOWED_TABLES = frozenset({"orders", "order_items", "products", "users"})

# Node types that must never appear in a read-only query. ``Command`` catches
# statements sqlglot does not model explicitly (e.g. TRUNCATE, GRANT, CALL).
_FORBIDDEN_NODES = tuple(
    node
    for node in (
        getattr(exp, name, None)
        for name in (
            "Insert",
            "Update",
            "Delete",
            "Merge",
            "Create",
            "Drop",
            "Alter",
            "AlterTable",
            "TruncateTable",
            "Grant",
            "Command",
            "Set",
        )
    )
    if node is not None
)

# Top-level statement shapes we accept (a plain SELECT, a CTE-led SELECT — which
# sqlglot still models as a Select — or a set operation of SELECTs).
_SELECT_SHAPES = tuple(
    node
    for node in (getattr(exp, name, None) for name in ("Select", "Union", "Intersect", "Except"))
    if node is not None
)


@dataclass(frozen=True)
class SqlValidation:
    """Result of validating a statement: ``ok`` plus the normalized SQL or an error."""

    ok: bool
    sql: str = ""
    error: str | None = None


def _expected_db_catalog(dataset: str) -> tuple[str | None, str | None]:
    """Split ``project.dataset`` (or ``dataset``) into (db, catalog) for comparison."""
    parts = [p for p in dataset.split(".") if p]
    if len(parts) >= 2:
        return parts[-1], parts[-2]  # db (dataset), catalog (project)
    if parts:
        return parts[0], None
    return None, None


def validate_select(
    sql: str,
    *,
    allowed_tables: frozenset[str] = DEFAULT_ALLOWED_TABLES,
    dataset: str = "bigquery-public-data.thelook_ecommerce",
    max_limit: int = 1000,
    dialect: str = "bigquery",
) -> SqlValidation:
    """Validate (and normalize) a generated query; see module docstring for the rules."""
    cleaned = (sql or "").strip().rstrip(";").strip()
    if not cleaned:
        return SqlValidation(ok=False, error="No SQL was generated.")

    try:
        statements = [s for s in sqlglot.parse(cleaned, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        return SqlValidation(ok=False, error=f"Could not parse SQL: {exc}")

    if len(statements) != 1:
        return SqlValidation(ok=False, error="Only a single SQL statement is allowed.")
    statement = statements[0]

    if not isinstance(statement, _SELECT_SHAPES):
        return SqlValidation(ok=False, error="Only read-only SELECT queries are allowed.")
    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return SqlValidation(ok=False, error="Only read-only SELECT queries are allowed.")

    allowed_lower = {name.lower() for name in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    expected_db, expected_catalog = _expected_db_catalog(dataset)
    for table in statement.find_all(exp.Table):
        name = (table.name or "").lower()
        if name in cte_names:
            continue  # a reference to a CTE, not a real table
        # Read-only INFORMATION_SCHEMA metadata views (table/column names + types) let the
        # agent answer database-structure questions (task.md). Allowed ONLY when scoped to
        # the configured dataset — they expose schema metadata, never row data or PII.
        if name.startswith("information_schema."):
            if (table.db or "").lower() != (expected_db or "").lower():
                return SqlValidation(
                    ok=False,
                    error="INFORMATION_SCHEMA queries must be scoped to dataset "
                    f"'{expected_db}' (e.g. `{dataset}`.INFORMATION_SCHEMA.COLUMNS).",
                )
            if (
                table.catalog
                and expected_catalog
                and table.catalog.lower() != expected_catalog.lower()
            ):
                return SqlValidation(
                    ok=False, error=f"Table must be in project '{expected_catalog}'."
                )
            continue
        if name not in allowed_lower:
            allowed = ", ".join(sorted(allowed_tables))
            return SqlValidation(
                ok=False,
                error=f"Query references a table outside the allowed set: '{table.name}'. "
                f"Only these tables may be queried: {allowed}.",
            )
        if table.db and expected_db and table.db.lower() != expected_db.lower():
            return SqlValidation(
                ok=False, error=f"Table '{table.name}' must be in dataset '{expected_db}'."
            )
        if table.catalog and expected_catalog and table.catalog.lower() != expected_catalog.lower():
            return SqlValidation(
                ok=False, error=f"Table '{table.name}' must be in project '{expected_catalog}'."
            )

    statement = _apply_limit(statement, max_limit)
    return SqlValidation(ok=True, sql=statement.sql(dialect=dialect))


def _apply_limit(statement: exp.Expression, max_limit: int) -> exp.Expression:
    """Inject a LIMIT when absent, or clamp a too-large numeric LIMIT (SELECT only)."""
    if not isinstance(statement, exp.Select):
        return statement  # set operations are left as-is; cost is still capped at execution
    limit_node = statement.args.get("limit")
    if limit_node is None:
        return statement.limit(max_limit)
    value = limit_node.expression
    if isinstance(value, exp.Literal) and value.name.isdigit() and int(value.name) > max_limit:
        return statement.limit(max_limit)
    return statement
