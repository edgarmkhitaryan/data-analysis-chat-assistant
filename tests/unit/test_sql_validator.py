"""Unit tests for the sqlglot read-only SQL validator (plan/007 §2, plan/011 §1.1).

Pure logic — no BigQuery, no network. Covers: accept valid SELECT/CTE; reject
DML/DDL, multi-statement, disallowed tables, wrong dataset; LIMIT inject/clamp.
"""

from assistant.safety.sql_validator import validate_select

DS = "bigquery-public-data.thelook_ecommerce"
ORDERS = f"`{DS}.orders`"
ITEMS = f"`{DS}.order_items`"
PRODUCTS = f"`{DS}.products`"


# --- Accept valid read-only queries ------------------------------------------


def test_accepts_simple_select():
    result = validate_select(f"SELECT status, COUNT(*) AS n FROM {ORDERS} GROUP BY status")
    assert result.ok
    assert result.error is None


def test_accepts_join_and_qualified_tables():
    sql = (
        f"SELECT p.category, SUM(oi.sale_price) AS revenue "
        f"FROM {ITEMS} oi JOIN {PRODUCTS} p ON oi.product_id = p.id "
        f"GROUP BY p.category ORDER BY revenue DESC LIMIT 10"
    )
    result = validate_select(sql)
    assert result.ok
    assert "limit 10" in result.sql.lower()


def test_accepts_cte_without_flagging_cte_name_as_table():
    sql = (
        f"WITH rev AS (SELECT product_id, SUM(sale_price) AS r FROM {ITEMS} GROUP BY product_id) "
        f"SELECT * FROM rev ORDER BY r DESC"
    )
    result = validate_select(sql)
    assert result.ok, result.error


def test_accepts_unqualified_allowed_table():
    result = validate_select("SELECT COUNT(*) AS n FROM orders")
    assert result.ok


# --- Reject writes / DDL / multi-statement -----------------------------------


def test_rejects_delete():
    result = validate_select(f"DELETE FROM {ORDERS} WHERE status = 'Cancelled'")
    assert not result.ok
    assert "read-only" in result.error.lower()


def test_rejects_insert_update_drop_truncate():
    for sql in (
        f"INSERT INTO {ORDERS} (id) VALUES (1)",
        f"UPDATE {ORDERS} SET status = 'x'",
        f"DROP TABLE {ORDERS}",
        f"TRUNCATE TABLE {ORDERS}",
    ):
        result = validate_select(sql)
        assert not result.ok, f"should reject: {sql}"


def test_rejects_stacked_statements():
    result = validate_select(f"SELECT 1 FROM {ORDERS}; DROP TABLE {ORDERS}")
    assert not result.ok
    assert "single" in result.error.lower()


def test_rejects_unparseable_sql():
    result = validate_select("SELECT FROM WHERE GROUP")
    assert not result.ok


def test_rejects_empty():
    assert not validate_select("   ").ok


# --- Table allow-list --------------------------------------------------------


def test_rejects_table_outside_allowlist():
    result = validate_select(f"SELECT * FROM `{DS}.payments`")
    assert not result.ok
    assert "allowed" in result.error.lower()


def test_rejects_wrong_dataset():
    result = validate_select("SELECT * FROM `bigquery-public-data.other_ds.orders`")
    assert not result.ok
    assert "dataset" in result.error.lower()


# --- INFORMATION_SCHEMA metadata (DB-structure questions) ---------------------


def test_accepts_information_schema_columns_scoped_to_dataset():
    result = validate_select(
        f"SELECT table_name, column_name, data_type FROM `{DS}`.INFORMATION_SCHEMA.COLUMNS"
    )
    assert result.ok, result.error


def test_accepts_information_schema_tables_unbackticked():
    result = validate_select(f"SELECT table_name FROM {DS}.INFORMATION_SCHEMA.TABLES")
    assert result.ok, result.error


def test_rejects_information_schema_of_other_dataset():
    result = validate_select(
        "SELECT * FROM `bigquery-public-data.other_ds`.INFORMATION_SCHEMA.COLUMNS"
    )
    assert not result.ok
    assert "dataset" in result.error.lower()


def test_information_schema_is_still_read_only():
    # The metadata exception must not become a DML/DDL loophole.
    result = validate_select(f"DROP TABLE `{DS}`.INFORMATION_SCHEMA.COLUMNS")
    assert not result.ok


# --- LIMIT injection / clamp -------------------------------------------------


def test_injects_limit_when_absent():
    result = validate_select(f"SELECT id FROM {ORDERS}", max_limit=1000)
    assert result.ok
    assert "limit 1000" in result.sql.lower()


def test_clamps_limit_over_cap():
    result = validate_select(f"SELECT id FROM {ORDERS} LIMIT 999999", max_limit=1000)
    assert result.ok
    assert "limit 1000" in result.sql.lower()
    assert "999999" not in result.sql


def test_preserves_limit_under_cap():
    result = validate_select(f"SELECT id FROM {ORDERS} LIMIT 5", max_limit=1000)
    assert result.ok
    assert "limit 5" in result.sql.lower()
