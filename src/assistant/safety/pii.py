"""Deterministic PII protection (plan/007 §3).

Masking happens *between BigQuery and the model*: the LLM — and therefore the
user — never sees raw PII, so leakage is impossible-by-construction rather than
discouraged-by-prompt. There is **no LLM involvement here and PII is never
mentioned in any prompt**; this is the security control, and it is the only one
we trust for PII.

Detection is layered:

1. **Schema-driven (primary):** columns named in the configured PII registry
   (``Settings.pii_mask_columns``) are masked by name — authoritative and exact.
2. **Pattern-driven (safety net):** regex for emails and phone numbers scans
   *every* string cell, catching PII that arrives via aliases, concatenations, or
   unexpected columns.

The same regex pass powers the **output guard** (:func:`scan_text`): a final scan
over the generated report text — the alarm that fires if anything PII-shaped ever
slips through to the model's output (see :mod:`assistant.agent.nodes.synthesize`).

Two masking styles (``Settings.pii_mask_style``):

- ``partial`` (default) — keep the value readable/distinguishable while hiding it
  (``jane.doe@example.com`` -> ``j***@e***.com``);
- ``redact`` — replace with an opaque token (``[REDACTED_EMAIL]``).
"""

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

# --- Detection patterns (the safety net) -------------------------------------

# Standard email shape.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Phone-shaped: an optional ``+country`` code, then 3-3-4 digit groups separated
# by spaces, dots, or hyphens (the area code may be parenthesized). We *require*
# the separators so we never mask plain integers like revenue or row counts, and
# the lookarounds keep us from matching digits embedded in a longer number (dates,
# IDs, ISO timestamps).
_PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)")


# --- Partial-mask primitives -------------------------------------------------


def _mask_email_value(value: str) -> str:
    """``jane.doe@example.com`` -> ``j***@e***.com`` (first char of local + domain)."""
    local, _, domain = value.partition("@")
    name, _, tld = domain.rpartition(".")
    masked_local = (local[:1] or "") + "***"
    # Keep the first char of the domain name + the TLD; degrade gracefully if the
    # domain has no dot.
    masked_domain = f"{name[:1]}***.{tld}" if name else (domain[:1] or "") + "***"
    return f"{masked_local}@{masked_domain}"


def _mask_phone_match(match: re.Match) -> str:
    """Keep the leading group + last four digits, mask the rest, preserve separators.

    ``+1-415-555-0199`` -> ``+1-***-***-0199``.
    """
    raw = match.group(0)
    digit_index = [i for i, ch in enumerate(raw) if ch.isdigit()]
    keep = set(digit_index[-4:])  # last four digits
    prev = None  # plus the leading contiguous digit run (country/area code)
    for i in digit_index:
        if prev is None or i == prev + 1:
            keep.add(i)
            prev = i
        else:
            break
    return "".join("*" if (ch.isdigit() and i not in keep) else ch for i, ch in enumerate(raw))


def _mask_freeform(value: str) -> str:
    """Keep the first character, mask the rest, preserve spaces.

    ``742 Evergreen Terrace`` -> ``7** ********* *******`` (also used for postal codes).
    """
    if not value:
        return value
    return value[0] + "".join(" " if ch == " " else "*" for ch in value[1:])


# --- Column maskers (schema-driven) ------------------------------------------


def _column_email(value: Any, style: str) -> str:
    if style == "redact":
        return "[REDACTED_EMAIL]"
    text = str(value)
    return _mask_email_value(text) if "@" in text else _mask_freeform(text)


def _column_geo(value: Any, style: str) -> str:
    """Coarsen precise geo to the integer degree (``37.7749`` -> ``37.x``).

    Non-numeric geo (e.g. a WKT ``user_geom`` like ``POINT(-122.4 37.8)``) is
    redacted wholesale, since it cannot be safely coarsened.
    """
    if style == "redact":
        return "[REDACTED_GEO]"
    try:
        return f"{int(float(value))}.x"
    except (TypeError, ValueError):
        return "[REDACTED_GEO]"


def _column_freeform(value: Any, style: str) -> str:
    if style == "redact":
        return "[REDACTED]"
    return _mask_freeform(str(value))


# Map a registry column name to how it is masked. Columns in the registry but not
# named here (e.g. a future ``ssn``) fall back to the freeform masker.
_COLUMN_MASKERS: dict[str, Callable[[Any, str], str]] = {
    "email": _column_email,
    "street_address": _column_freeform,
    "postal_code": _column_freeform,
    "latitude": _column_geo,
    "longitude": _column_geo,
    "user_geom": _column_geo,
}


# --- The regex safety-net pass (rows + output guard) -------------------------


def _scan(text: str, style: str) -> tuple[str, int]:
    """Mask every email/phone in ``text``; return (cleaned_text, hit_count)."""
    hits = 0

    def email_repl(match: re.Match) -> str:
        nonlocal hits
        hits += 1
        return "[REDACTED_EMAIL]" if style == "redact" else _mask_email_value(match.group(0))

    def phone_repl(match: re.Match) -> str:
        nonlocal hits
        hits += 1
        return "[REDACTED_PHONE]" if style == "redact" else _mask_phone_match(match)

    text = _EMAIL_RE.sub(email_repl, text)
    text = _PHONE_RE.sub(phone_repl, text)
    return text, hits


# --- Public API --------------------------------------------------------------


def mask_rows(
    rows: Iterable[Mapping[str, Any]],
    mask_columns: Iterable[str],
    style: str = "partial",
) -> tuple[list[dict[str, Any]], int]:
    """Mask PII in query result rows before they reach the model.

    Registry columns are masked by name (primary); every other *string* cell is
    scanned for email/phone patterns (safety net). Each masked cell increments the
    returned count (a metric: ``pii_masked_count``).

    Args:
        rows: JSON-safe result rows (from ``execute_sql``).
        mask_columns: the PII registry (``Settings.pii_mask_columns``).
        style: ``"partial"`` or ``"redact"``.

    Returns:
        ``(masked_rows, masked_count)``.
    """
    columns = set(mask_columns)
    masked_rows: list[dict[str, Any]] = []
    masked_count = 0

    for row in rows:
        new_row: dict[str, Any] = {}
        for key, value in row.items():
            if key in columns and value is not None:
                masker = _COLUMN_MASKERS.get(key, _column_freeform)
                masked = masker(value, style)
                new_row[key] = masked
                if masked != value:
                    masked_count += 1
            elif isinstance(value, str):
                cleaned, hits = _scan(value, style)
                new_row[key] = cleaned
                if hits:
                    masked_count += 1
            else:
                new_row[key] = value
        masked_rows.append(new_row)

    return masked_rows, masked_count


def scan_text(text: str, style: str = "partial") -> tuple[str, int]:
    """Output guard: re-scan generated report text for any email/phone that leaked.

    Returns ``(cleaned_text, leak_count)``. A non-zero count is a bug to fix — see
    the ``pii_leak_prevented`` safety event in
    :mod:`assistant.agent.nodes.synthesize`.
    """
    if not text:
        return text, 0
    return _scan(text, style)
