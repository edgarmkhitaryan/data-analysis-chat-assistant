"""Schema provider node: builds the database context injected into SQL generation.

It combines the *authoritative* column list (read live from BigQuery) with
hand-written **business annotations** — join keys, what "revenue"/"profit" mean,
which columns are PII — so the model generates correct, business-aware SQL rather
than guessing semantics from column names. The result is cached per dataset
(the schema is effectively static), so we pay the metadata reads once per process.
"""

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState
from assistant.bigquery import BigQueryRunner

_TABLES = ("orders", "order_items", "products", "users")

_TABLE_NOTES = {
    "orders": (
        "One row per customer order. Join to order_items on order_id; "
        "user_id references users.id. created_at is the order date; status is the "
        "fulfillment state (e.g. Complete, Shipped, Processing, Cancelled, Returned)."
    ),
    "order_items": (
        "One row per item sold — the revenue grain. sale_price is the realized "
        "revenue per item; total revenue = SUM(sale_price). product_id references "
        "products.id; order_id references orders.order_id; user_id references users.id."
    ),
    "products": (
        "Product catalog. retail_price is list price and cost is unit cost, so "
        "profit/margin = order_items.sale_price - products.cost. category, brand, "
        "and department describe the product."
    ),
    "users": (
        "Customers. Contains PII (email, street_address, postal_code, latitude, "
        "longitude) — never expose raw PII in answers. Non-sensitive attributes "
        "useful for analysis: age, gender, city, state, country, traffic_source."
    ),
}

_BUSINESS_RULES = """\
Business rules:
- "Revenue"/"sales" = SUM(order_items.sale_price).
- "Profit"/"margin" = SUM(order_items.sale_price - products.cost), joining
  order_items to products on order_items.product_id = products.id.
- "Top"/"best" products or customers means by revenue unless stated otherwise.
- Time periods use created_at; treat the data as historical (do not assume "today")."""

_SQL_CONVENTIONS = """\
SQL conventions (BigQuery Standard SQL):
- SELECT statements only. Fully-qualify tables as
  `bigquery-public-data.thelook_ecommerce.<table>`.
- Add a sensible LIMIT (<= 1000) unless the question needs a full aggregate.
- Prefer explicit JOINs and GROUP BY; give aggregates readable column aliases."""

_cache: dict[str, str] = {}


def build_schema_context(runner: BigQueryRunner) -> str:
    """Return the cached schema context for the runner's dataset, building it once."""
    if runner.dataset_id not in _cache:
        _cache[runner.dataset_id] = _format_context(runner)
    return _cache[runner.dataset_id]


def _format_context(runner: BigQueryRunner) -> str:
    blocks = [
        f"You are querying the BigQuery dataset `{runner.dataset_id}`, "
        "the data of a fictional retail e-commerce company.",
        "",
        "Tables:",
    ]
    for table in _TABLES:
        columns = runner.get_table_schema(table)
        column_list = ", ".join(f"{col.name} ({col.type})" for col in columns)
        blocks += [
            f"\n### {table}",
            _TABLE_NOTES[table],
            f"columns: {column_list}",
        ]
    blocks += ["", _BUSINESS_RULES, "", _SQL_CONVENTIONS]
    return "\n".join(blocks)


def get_schema(state: AgentState, deps: AgentDeps) -> dict:
    """Populate ``schema_context`` for downstream SQL generation."""
    return {"schema_context": build_schema_context(deps.runner)}
