"""Unit tests for deterministic PII masking (plan/007 §3, plan/011 §1.1).

Pure logic — no network, no LLM, no settings load. Covers schema-driven column
masking, the email/phone regex safety-net, the output guard, mask-count accuracy,
and the headline guarantee: no raw PII survives into the masked output.
"""

from types import SimpleNamespace

from assistant.agent.nodes.mask_pii import mask_pii
from assistant.safety.pii import mask_rows, scan_text

# The default registry (mirrors Settings.pii_mask_columns).
REGISTRY = ["email", "street_address", "postal_code", "latitude", "longitude", "user_geom"]

# A realistic users row with every sensitive field populated.
RAW_USER = {
    "id": 4242,
    "first_name": "Jane",
    "last_name": "Doe",
    "email": "jane.doe@example.com",
    "age": 34,
    "gender": "F",
    "city": "Springfield",
    "state": "Illinois",
    "country": "United States",
    "street_address": "742 Evergreen Terrace",
    "postal_code": "90210",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "user_geom": "POINT(-122.4194 37.7749)",
    "traffic_source": "Search",
}


# --- Schema-driven column masking (partial style) ----------------------------


def test_email_column_partial_masked():
    [row], count = mask_rows([RAW_USER], REGISTRY)
    assert row["email"] == "j***@e***.com"
    assert count >= 1


def test_address_and_postal_masked():
    [row], _ = mask_rows([RAW_USER], REGISTRY)
    assert row["street_address"] == "7** ********* *******"
    assert row["postal_code"] == "9****"


def test_geo_coarsened():
    [row], _ = mask_rows([RAW_USER], REGISTRY)
    assert row["latitude"] == "37.x"
    assert row["longitude"] == "-122.x"
    # A WKT geometry string cannot be coarsened numerically -> redacted wholesale.
    assert row["user_geom"] == "[REDACTED_GEO]"


def test_non_pii_columns_left_visible():
    """Names and coarse geography stay visible so reports can name top customers."""
    [row], _ = mask_rows([RAW_USER], REGISTRY)
    assert row["first_name"] == "Jane"
    assert row["last_name"] == "Doe"
    assert row["city"] == "Springfield"
    assert row["state"] == "Illinois"
    assert row["age"] == 34
    assert row["id"] == 4242


# --- The headline guarantee: no raw PII survives -----------------------------


def test_no_raw_pii_survives():
    """Nothing a trace/log would persist (the masked row) contains a raw value."""
    [row], _ = mask_rows([RAW_USER], REGISTRY)
    blob = repr(row)
    for raw in ("jane.doe@example.com", "742 Evergreen Terrace", "37.7749", "-122.4194"):
        assert raw not in blob


# --- Mask count accuracy -----------------------------------------------------


def test_masked_count_counts_each_masked_cell():
    # email, street_address, postal_code, latitude, longitude, user_geom = 6 cells.
    _, count = mask_rows([RAW_USER], REGISTRY)
    assert count == 6


def test_clean_row_masks_nothing():
    clean = {"category": "Jeans", "revenue": 12345.67, "orders": 1000}
    [row], count = mask_rows([clean], REGISTRY)
    assert row == clean
    assert count == 0


def test_none_values_untouched():
    [row], count = mask_rows([{"email": None, "city": None}], REGISTRY)
    assert row == {"email": None, "city": None}
    assert count == 0


# --- Pattern-driven safety net (PII via unexpected columns) -------------------


def test_email_in_unregistered_column_caught_by_regex():
    rows = [{"notes": "reach me at john@acme.io for details"}]
    [row], count = mask_rows(rows, REGISTRY)
    assert "john@acme.io" not in row["notes"]
    assert "j***@a***.io" in row["notes"]
    assert count == 1


def test_phone_in_unregistered_column_masked_keeping_last_four():
    rows = [{"note": "call +1-415-555-0199 today"}]
    [row], count = mask_rows(rows, REGISTRY)
    assert "+1-415-555-0199" not in row["note"]
    assert "+1-***-***-0199" in row["note"]
    assert count == 1


def test_plain_numbers_are_not_treated_as_phones():
    """Revenue, counts, years and ISO dates must not be mangled as phone numbers."""
    rows = [{"label": "revenue 1234567 in 2024-01-15 across 4155550199 rows"}]
    [row], count = mask_rows(rows, REGISTRY)
    assert row["label"] == rows[0]["label"]
    assert count == 0


# --- Redact style ------------------------------------------------------------


def test_redact_style_uses_tokens():
    [row], _ = mask_rows([RAW_USER], REGISTRY, style="redact")
    assert row["email"] == "[REDACTED_EMAIL]"
    assert row["street_address"] == "[REDACTED]"
    assert row["latitude"] == "[REDACTED_GEO]"


# --- Output guard (the last line) --------------------------------------------


def test_output_guard_catches_leaked_email():
    text = "Our top customer is jane.doe@example.com with $5,000 in sales."
    cleaned, leaks = scan_text(text)
    assert "jane.doe@example.com" not in cleaned
    assert "j***@e***.com" in cleaned
    assert leaks == 1


def test_output_guard_passes_clean_text():
    text = "Top category was Jeans at $1,234,567.00 in revenue."
    cleaned, leaks = scan_text(text)
    assert cleaned == text
    assert leaks == 0


def test_output_guard_empty_text():
    assert scan_text("") == ("", 0)


# --- Node wiring -------------------------------------------------------------


def test_mask_pii_node_returns_masked_rows_and_count():
    deps = SimpleNamespace(
        settings=SimpleNamespace(pii_mask_columns=REGISTRY, pii_mask_style="partial")
    )
    state = {"raw_rows": [RAW_USER]}
    out = mask_pii(state, deps)
    assert out["pii_masked_count"] == 6
    assert out["masked_rows"][0]["email"] == "j***@e***.com"
    # The node never echoes raw_rows back into state.
    assert "raw_rows" not in out


def test_mask_pii_node_handles_no_rows():
    deps = SimpleNamespace(
        settings=SimpleNamespace(pii_mask_columns=REGISTRY, pii_mask_style="partial")
    )
    out = mask_pii({}, deps)
    assert out == {"masked_rows": [], "pii_masked_count": 0}
