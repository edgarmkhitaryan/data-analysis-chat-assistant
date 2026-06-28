# High-Level Design & Technical Explanation

**Data Analysis Chat Assistant** — a conversational agent that turns a retail manager's natural-language
question into validated, read-only SQL over BigQuery, executes it, and returns an analyst-grade report in
the manager's preferred style, while staying safe, resilient, observable, and self-improving.

This document is **Deliverable 2**, the detailed technical explanation. It is organized to map directly
onto the five sub-parts the brief asks for:

| Brief asks for | Section |
|---|---|
| 2.1 Reasoning for chosen cloud services / LLM / frameworks | [§3](#3-reasoning-for-the-chosen-cloud-llm-and-frameworks) |
| 2.2 Data flow between components | [§4](#4-data-flow-between-components) |
| 2.3 Error handling & fallback strategies | [§5](#5-error-handling--fallback-strategies) |
| 2.4 Setup instructions & example run | [§6](#6-setup--example-run) |
| 2.5 How each of the 8 requirements is solved | [§7](#7-how-each-requirement-is-solved) |

Diagrams live in **[architecture.md](architecture.md)**; how to run it is in the **[README](../README.md)**.

> **Production-first, faithfully prototyped.** The design below is the *production* system. The prototype
> in this repo implements the **same LangGraph control flow and the same trust boundaries**, swapping
> managed GCP services for local equivalents (in-process vector store, SQLite, local files). Where the
> prototype deliberately simplifies, this document says so explicitly rather than implying production
> scale — see [§9, Limitations & honesty](#9-limitations--honesty).

---

## 1. The problem, and what is actually hard

Non-technical executives (Store and Regional Managers) need trustworthy, analyst-grade answers from raw
transaction data — *"Why is the X branch underperforming, and how does it compare to Y?"* — without
writing SQL. The brief is explicit that assessment focuses on **system design, the technical explanation,
and an elegant prototype**, not feature breadth.

Text-to-SQL is the easy part; it is a solved-enough building block. The genuine difficulty is everything
*around* it, and the architecture is shaped by those four pressures:

1. **Hybrid intelligence** — reuse how analysts previously *interpreted* questions (the Golden Bucket),
   not just generate SQL from a schema.
2. **Safety** — never leak customer PII, never run a destructive action without explicit consent, refuse
   non-analysis / malicious input.
3. **Trust** — self-correct on errors, degrade gracefully, and be measurable so we know when and why it
   fails.
4. **Operability** — let non-developers change tone weekly, and let the system learn from past
   interactions, with no redeploy and no retraining.

The whole system therefore treats text-to-SQL as *one node* in a larger, well-instrumented graph.

---

## 2. Architecture & components

The system is a stateful conversational agent fronted by a thin client, orchestrated as a **LangGraph**
graph, grounded by two knowledge sources (**BigQuery** for facts, the **Golden Bucket** for
interpretation), and wrapped in cross-cutting layers for safety, memory/persona, resilience, and
observability. See the [production architecture diagram](architecture.md#1-production-system-architecture)
and the [as-built LangGraph topology](architecture.md#2-the-agent-graph-langgraph-topology--as-built).

**Logical building blocks and their responsibilities:**

| Block | Responsibility | Prototype location |
|---|---|---|
| **Client / Channel** | Chat I/O; resumes confirmation interrupts; renders reports/traces. Holds **no business logic**. | `src/assistant/cli.py` |
| **Agent Orchestrator** | The brain: contextualize → guard → route → (analysis \| reports \| preference); compound decompose/synthesize; self-correction; report synthesis; human-in-the-loop confirm. | `src/assistant/agent/` |
| **LLM & Embeddings** | Chat (SQL + report generation) and question embeddings, behind a factory with two-tier routing. | `src/assistant/llm/` |
| **Data Warehouse** | Read-only BigQuery access with a dry-run cost guard. | `src/assistant/bigquery/runner.py` |
| **Golden Bucket** | Vector retrieval of analyst Trios; the automatic learning loop that grows it. | `src/assistant/golden/`, `src/assistant/memory/feedback.py` |
| **Safety layer** | Input guard + injection filter, read-only SQL validation, PII masking + output guard, owner-scoped oversight. | `src/assistant/safety/`, `src/assistant/reports/` |
| **Memory & Persona** | Per-user free-form preferences; org persona/tone config (hot-reloaded). | `src/assistant/memory/profiles.py`, `src/assistant/persona/loader.py` |
| **Resilience** | Retries + circuit breaker for both external dependencies; bounded self-correction. | `src/assistant/resilience.py`, `src/assistant/llm/`, `src/assistant/bigquery/` |
| **Observability** | Per-turn run traces, agent-level metrics, structured JSON logs. | `src/assistant/observability/` |
| **Quality Assurance** | Offline eval harness (objective checks + LLM-as-judge) + a test pyramid. | `src/assistant/eval/`, `tests/` |
| **Configuration** | One typed settings surface; two required secrets, sensible defaults for the rest. | `src/assistant/config.py`, `.env.example` |

A single typed **`AgentState`** (`src/assistant/agent/state.py`, a `TypedDict`) flows through the graph;
LangGraph merges each node's returned delta into it. Two design choices in the state pay off everywhere:

- **`raw_question` vs `question`** makes contextualization auditable — the trace shows exactly what the
  user typed *and* what the system resolved it to before any SQL was generated.
- **`raw_rows` vs `masked_rows`** makes the PII boundary explicit in the type system — the report node
  reads `masked_rows` only.

---

## 3. Reasoning for the chosen cloud, LLM, and frameworks

Decisions are recorded as ADRs (decision → why → trade-off). The full service table with per-service
rationale is in [architecture.md §5](architecture.md#5-why-each-service-the-rationale); the load-bearing
choices:

### 3.1 Orchestration — LangGraph (+ LangChain Core)
The workflow is intrinsically a **graph with a loop and a pause**: a bounded self-correction loop around
SQL, and a human-in-the-loop confirmation pause before destructive deletes. LangGraph models loops,
conditional routing, and **`interrupt()`** as first-class concepts, and its **checkpointer** gives durable,
resumable conversation state — exactly what both the "discuss about it" requirement and the oversight flow
need. Alternatives (a plain LangChain AgentExecutor, a raw function-calling loop, LlamaIndex) would have us
re-implement checkpointing, routing, and interrupts by hand. **Trade-off:** a little more upfront structure
(explicit state + nodes) in exchange for deterministic, inspectable, resumable control flow.

### 3.2 Cloud — GCP-native (data gravity)
The warehouse (BigQuery) and the recommended model (Gemini) already live on GCP. Co-locating compute
(**Cloud Run**), model + embeddings (**Vertex AI**), vector search (**Vertex AI Vector Search**),
operational state (**Cloud SQL**), and identity in one IAM/trust boundary minimizes egress, latency, and
credential sprawl, and lets read-only access be enforced by **IAM**, not just by prompt.

### 3.3 LLM — Google Gemini, two-tier
Recommended by the brief; strong SQL/reasoning; the **same credential provides embeddings**. We run a
**two-tier routing** policy (`src/assistant/llm/client.py`, `get_chat_model(cheap=…)`):

- **Main model** (`LLM_MODEL`, default `gemini-3.1-flash-lite`) for the four quality-critical calls:
  `generate_sql`, `report`, compound `synthesize`, and the eval `judge`.
- **Cheap model** (`LLM_MODEL_CHEAP`, default `gemini-2.5-flash-lite`, ≈2.5× cheaper input / ≈3.75×
  cheaper output) for the five low-stakes *structured* calls: intent guard, contextualize, decompose,
  preference-merge, report-command parse.

This cuts cost and — because Gemini free-tier limits are **per model** — spreads load across two quota
buckets, easing 429s. **Why `gemini-3.1-flash-lite` as the default:** during development the
`gemini-2.5-flash` free tier was capped at ~20 requests/day, which blocked iteration and would break a
reviewer-run demo; `3.1-flash-lite` has a more generous bucket. Both are config — a one-line change. In
production we move to **Vertex AI** for IAM auth (no raw keys), higher quotas, and data-residency control.
The provider is abstracted behind the `llm/` factory, so swapping to OpenRouter/Ollama is a config change.

### 3.4 Knowledge — RAG over fine-tuning
Analyst interpretation changes continuously. Retrieval lets us change behavior by **adding a Trio** — no
retraining, and full traceability of *why* an answer was shaped a certain way (you can see which Trio
informed a report). Embeddings are `gemini-embedding-001`; the prototype uses an in-process NumPy cosine
index behind the retriever interface, mapping mechanically to Vertex AI Vector Search.

### 3.5 Stores & config
**Cloud SQL (Postgres)** for Saved Reports, profiles, and the LangGraph checkpointer — we need
transactional deletes, per-user ownership, and durable conversation state in one place (SQLite locally).
**Persona/tone is externalized config** (YAML locally; GCS/Firestore in prod), hot-reloaded so a
non-developer changes it weekly with no redeploy. **Secret Manager** in production; `.env` locally.

---

## 4. Data flow between components

Four flows cover the system. The first is drawn as a sequence diagram in
[architecture.md §3](architecture.md#3-data-flow--the-analysis-happy-path-with-self-correction).

### 4.1 Analysis question (happy path)
1. **Contextualize** — rewrite the turn into a *standalone* question using thread history + timestamps
   (resolving "that"/"it"/relative dates). First turn / empty history is a passthrough with **no LLM
   cost**. A rule-based **injection filter** runs here first, before the rewrite LLM. If the rewrite is
   below a confidence floor (`CONTEXTUALIZE_CONFIDENCE_FLOOR`, default 0.6), route to **clarify** and ask
   one targeted question instead of guessing.
2. **Guard** — classify intent (`analysis | manage_reports | update_preference | rejected`) with the cheap
   model, told **not to default to analysis** (so a bare "hi" is refused). The injection filter runs again
   here. Off-topic / malicious turns are refused *before* any expensive work.
3. **Load context** — read the manager's persona + free-form preferences for this turn.
4. **Decompose** — detect a **compound** ask (e.g. *"top products by revenue **and** compare X to Y"*) and
   split into ≤`MAX_SUB_QUESTIONS` (default 4) self-contained sub-questions. A single question is the
   one-element case; both flow through the same `run_compound` path.
5. **Per sub-question, the analysis pipeline** (a compiled subgraph): **retrieve** top-k Trios → **schema**
   (cached) → **generate SQL** → **validate** (read-only, allow-list, LIMIT) → **dry-run cost guard** →
   **execute** → (**self-correct** on error/empty, bounded) → **mask PII** → **synthesize report**.
6. **Synthesize** — merge sub-reports into one briefing, preserving each section's figures verbatim and
   labeling any failed sub-question (partial-failure tolerant).
7. **Instrument & learn** — persist the trace, update metrics, and (after responding) run the automatic
   learning gate on a background thread.

### 4.2 Report management (destructive) flow
Intent `manage_reports` → **parse** into `{action, filters}`. **Save / list / view** are non-destructive
and execute directly (owner-scoped). **Delete** → **resolve targets** scoped to the requesting manager →
if matches, hit a LangGraph **`interrupt()`** that returns a blast-radius summary and **pauses** (nothing
mutated) → on explicit **confirm**, delete and **audit**; on anything else, cancel safely. A delete with
no qualifier matches nothing and asks the user to be specific — it never defaults to "everything".

### 4.3 Learning loop (asynchronous, automatic)
Each completed analysis turn emits a candidate (question, final SQL, final report, **masked rows**,
outcome signals). A three-stage gate — **deterministic metrics → novelty/dedup → LLM-as-judge
(faithfulness AND intent-satisfaction)** — decides promotion, with **no human in the loop**. Approved
candidates become `source="learned"` Trios, embedded and upserted, retrievable immediately and on next
start. Diagram: [architecture.md §4](architecture.md#4-the-automatic-learning-loop).

### 4.4 User-preference update (per-manager memory)
Every turn **reads** the profile (the *read* path, applied at synthesis). A **standing** preference
("from now on use tables") is detected by the guard and **persisted synchronously** by `update_prefs`
(merged into the existing free-form description). A **combined** "set preference + ask a question" turn
persists the preference **and** continues into the analysis with it applied — the question is never
dropped. A **one-off** aside ("…as bullets just this once") applies to that turn only and is not persisted.

---

## 5. Error handling & fallback strategies

Goal: **detect SQL errors and empty results and self-correct; survive third-party outages; never inflate
costs; never crash the UI.** Each is a distinct mechanism, and resilience wraps **both** external
dependencies (Gemini and BigQuery) through one layer (`src/assistant/resilience.py`).

### 5.1 Bounded self-correction (the headline behavior)
A loop in the graph (`generate_sql → validate_sql → execute_sql → self_correct → generate_sql`), bounded
by `MAX_SQL_ATTEMPTS` (default 3):
- **Validation failure** (bad syntax, disallowed statement/table, over-budget) → the *specific* validator
  message is fed back for a corrected query; the query never runs.
- **Execution error** → BigQuery's exact error string (e.g. `Unrecognized name: revenue`) is fed back as
  targeted repair context.
- **Empty result** → treated as a soft failure: **one** guided reformulation (reconsider date/status/
  narrow predicates), then an **honest "no data"** rather than fabrication.

Because it is a graph loop, not prompt recursion, every attempt is an inspectable, metered state
transition and the cost is hard-capped.

### 5.2 Resilience to third-party failures
`resilient_call` wraps each dependency with **tenacity** (exponential backoff + jitter,
`LLM_MAX_RETRIES`=4 attempts, base 1s capped at 8s) and a **circuit breaker** (`CIRCUIT_BREAKER_THRESHOLD`
=5 consecutive transient failures → open; `CIRCUIT_BREAKER_COOLDOWN_SECONDS`=30 → half-open probe). A
classifier splits **transient** (429/5xx/timeout/connection — retried) from **permanent** (auth/4xx —
fail fast, no wasted spend). Gemini calls share one process-wide `"gemini"` breaker (so the dependency is
treated as a whole regardless of model tier); BigQuery has its own `"bigquery"` breaker. The model client
sets `max_retries=0` so tenacity is the **single** retry layer (no double-retry multiplication).

### 5.3 Graceful degradation (never crash, never fabricate)
Three "useful answer instead of failing" behaviors, earliest first: **clarify** (ask, don't guess — the
cheapest recovery, no BigQuery spend); **partial synthesis** (deliver the sub-questions that succeeded,
label the ones that didn't); and **graceful degradation** (loop exhausted / dependency down → a clear,
non-fabricating message of what was tried and a suggested reformulation). The CLI wraps every turn in a
top-level handler, so any unexpected exception becomes a friendly message + a `run_id` and the REPL stays
alive. The background learning loop is similarly isolated — it can never break a turn.

### 5.4 Cost control ("self-correct … without inflating costs")
Bounded iterations (`MAX_SQL_ATTEMPTS`, `MAX_SUB_QUESTIONS`); the breaker caps retry storms; cheap-first
two-tier routing; a **pre-flight dry-run cost guard** (`MAX_BYTES_BILLED` ≈ 2 GB) plus a hard
`maximum_bytes_billed` cap; mandatory `LIMIT` injection (`SQL_MAX_LIMIT`=1000); compact schema context +
top-k (not top-everything) Trios + truncated row samples; and **early refusal** of off-topic/malicious
turns before the expensive pipeline.

### 5.5 Error taxonomy → handling

| Failure | Detected by | Handling |
|---|---|---|
| Ambiguous follow-up | `contextualize` low confidence | clarify; no SQL spent |
| Malformed / non-SELECT SQL | `validate_sql` (sqlglot) | repair via self-correction; never executed |
| Over-budget query | dry-run cost guard | repair (narrow); never executed |
| BigQuery execution error | `execute_sql` exception | feed exact error back; bounded repair |
| Empty result set | `row_count == 0` | one guided retry, then honest "no data" |
| One sub-question fails (compound) | per-sub budget exhausted | synthesize the rest; failed section labeled |
| LLM transient (429/5xx/timeout) | `is_transient` in `resilience.py` | backoff + retry; breaker if persistent |
| LLM permanent (auth/4xx) | `is_transient` → false | fail fast → degrade, no retry |
| Embeddings/vector unavailable | retriever exception | proceed schema-only; log "cold retrieval" |
| Reports store error | `reports/store.py` | transactional; report failure, no partial delete |
| Any unhandled exception | CLI top-level handler | friendly message + run_id; REPL survives |

---

## 6. Setup & example run

Full, copy-pasteable setup is in the **[README](../README.md#setup)**. In brief:

```bash
python -m venv venv && source venv/bin/activate
make install                 # editable install + dev tools
cp .env.example .env         # then set GEMINI_API_KEY and GOOGLE_CLOUD_PROJECT
gcloud auth application-default login   # BigQuery access (ADC)
make check                   # smoke-test Gemini + BigQuery connectivity
make run                     # start the CLI
```

Only **two** secrets are required (`GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`); every other setting has a
sensible default, validated at startup by `src/assistant/config.py`. For an isolated environment, the repo
also ships a `Dockerfile` (`docker build -t assistant .` → `docker run`); the runtime image carries the
seed data, runs as a non-root user, and mounts your BigQuery ADC read-only — see
[README › Run with Docker](../README.md#run-with-docker-optional-isolated).

### Example run (illustrative of the CLI's actual rendering)

```text
you › Which product categories drove the most revenue?

  ┌─ SQL ─────────────────────────────────────────────────────────────┐
  │ SELECT p.category, SUM(oi.sale_price) AS revenue                    │
  │ FROM `bigquery-public-data.thelook_ecommerce.order_items` oi        │
  │ JOIN `bigquery-public-data.thelook_ecommerce.products` p            │
  │   ON oi.product_id = p.id                                           │
  │ WHERE oi.status NOT IN ('Cancelled','Returned')                     │
  │ GROUP BY p.category ORDER BY revenue DESC LIMIT 1000                │
  └────────────────────────────────────────────────────────────────────┘
  ┌─ Report ──────────────────────────────────────────────────────────┐
  │ **Bottom line:** Outerwear & Coats led revenue, followed by Jeans   │
  │ and Sweaters. ... (persona: Concise Executive; manager prefs applied)│
  └────────────────────────────────────────────────────────────────────┘
  run_id=ab12cd34ef56 · /trace for the step timeline

you › /trace
run_id=ab12cd34ef56  user=manager_a  thread=9f0a1b2c3d4e
raw_question="Which product categories drove the most revenue?"
header: question=Which product categories drove the most revenue?  intent=analysis
  └─ contextualize      history_used=False  (1.2ms)
  └─ guard_input        intent=analysis  (312.0ms)
  └─ decompose          is_compound=False  (180.4ms)
  └─ retrieve_golden    trio_ids=['trio_0004', 'trio_0005']  (140.6ms)
  └─ generate_sql       attempt=1  sql=SELECT p.category, SUM(oi.sale_price) ...  (1100.0ms)
  └─ validate_sql       ok=True  (3.1ms)
  └─ execute_sql        rows=11  (520.3ms)
  └─ mask_pii           pii_masked=0  (1.0ms)
  └─ synthesize_report  report_chars=820  (1400.0ms)
outcome: status=success  rows=11  total_ms=3658.6
```

> This reproduces the **format** the `/trace` renderer actually emits (the header line, the `  └─ node …
> (Nms)` step lines in milliseconds, and the `outcome:` line — see `observability/tracing.py`). The exact
> figures and the precise fields shown depend on the live run. The five-step
> **[demo script](../README.md#demo-script-the-five-moments)** in the README reproduces each requirement
> live, including a self-correction round-trip and the PII/oversight controls.

---

## 7. How each requirement is solved

One tight section per requirement: what the brief asks, how we solve it, where in code, and how a reviewer
can see it.

### Requirement 1 — Hybrid Intelligence (the Golden Bucket)
**Ask:** don't rely on SQL alone — use a bucket of analyst Trios (Question → SQL → Report) to apply prior
interpretation; explain how it is **updated** and how relevant data is **retrieved** at query time.

**Solution.** A **Trio** (`src/assistant/golden/models.py`) is a teaching example, not a cache: it shows
the model which tables, which revenue definition, which grouping, and what a good narrative looks like, and
the model *adapts* it to the new question. **Retrieval** (`golden/retriever.py`, `golden/store.py`): embed
the standalone question with `gemini-embedding-001`, take the **top-k** (default 3) Trios above a cosine
floor (default 0.68), and inject them into both the SQL-generation prompt (as `question → SQL` exemplars)
and the report prompt (as `question → report` style exemplars). If nothing clears the floor, we proceed
schema-only and log a **"cold retrieval"** — the precise signal that a new Trio is needed. A subtle but
important detail: we embed the **question only**, with an asymmetric `task_type` (stored Trios as
`RETRIEVAL_DOCUMENT`, live queries as `RETRIEVAL_QUERY`) so matches land on intent.

**Updating the bucket — two paths into one index:**
- **Curated authoring** — analysts add/edit JSON Trios (`data/golden_trios/` → GCS in prod);
  `scripts/ingest_golden.py` (`make ingest`) re-embeds. The index self-heals via a **content fingerprint**
  (sha256 of id + question), so it only rebuilds when Trios actually change.
- **Automatic learning loop** — see [Requirement 4](#requirement-4--continuous-improvement-the-learning-loop).

**Prototype ships:** 12 seed Trios spanning every expected capability (customer behavior, product
performance, time-based, comparative, DB-structure). **See it:** ask *"Which product categories generate
the most revenue?"* then `/trace` — the `retrieve_golden` line shows the Trio ids that grounded the answer.

### Requirement 2 — Safety & PII Masking
**Ask:** only answer analysis questions, resist malicious users, and **never display Customer Phones/Emails
even if the SQL retrieves them**.

**Solution — three boundaries, all in code, never by prompt.** (Diagram:
[architecture.md §7](architecture.md#7-the-pii-trust-boundary).)
1. **Input guard** (`safety/input_guard.py`, `agent/nodes/guard.py`): a **rule-based injection filter**
   (instruction-override, prompt-extraction, jailbreak, write-SQL patterns) runs *before any LLM call* —
   twice, in `contextualize` and `guard`, so a malicious *follow-up* can't slip past via the clarify
   branch. Intent is then classified by the cheap model. **Refusals are deterministic** — adversarial
   input is never fed back to an LLM; the decline is a fixed, graceful capability message.
2. **SQL validation** (`safety/sql_validator.py`, sqlglot AST): single statement, **SELECT-only** (any
   DML/DDL node anywhere → reject), a **four-table allow-list** (+ dataset-scoped `INFORMATION_SCHEMA`
   for DB-structure questions), and mandatory **`LIMIT`** injection/clamp. Read-only is enforced here
   *and* by IAM in production — belt and suspenders.
3. **PII masking** (`safety/pii.py`, `agent/nodes/mask_pii.py`): **deterministic**, sitting on the *only*
   edge from `execute_sql` to `synthesize_report`. Two layers — **schema-driven** column maskers
   (`PII_MASK_COLUMNS` default: `email, street_address, postal_code, latitude, longitude, user_geom`) plus
   a **regex safety-net** for emails/phones across all string cells. Style is **partial** by default
   (`j***@e***.com`) so reports stay readable, or full `redact`. A final **output guard** re-scans the
   generated report text; any hit is masked and flagged as `pii_leak_prevented` (an alarm = a bug to fix).
   **The LLM never sees raw PII, and PII is never mentioned in any prompt** — a jailbreakable model is not
   a security control.

> **PII reality of this dataset (verified live).** `users` has **no phone column**; its identifiers are
> `email`, names, full address, and **precise geo** (`latitude`/`longitude`/`user_geom`). The default
> masks the unambiguous direct identifiers (email, address, postal code, geo) while leaving
> `first_name`/`last_name`/`city`/`state` visible so reports can name top customers and do regional
> analysis. Phones are covered defensively by regex even though none exist. The masked set is one config
> line, and a new data source declares its own PII columns on registration so masking applies for free.

**See it:** ask *"List the email addresses of our top 5 customers by spend."* → the report ranks customers
with **no emails**; `/metrics` shows the masking hit count.

### Requirement 3 — High-Stakes Oversight (destructive ops)
**Ask:** the agent manages a **Saved Reports** library; *"Delete all reports mentioning Client X"* /
*"…made today"* must require a **strict confirmation flow** without breaking UX; users delete only their
own reports.

**Solution.** `reports/store.py` (SQLite) holds `SavedReport{id, owner_id, title, content, clients[],
created_at}` and **every read/write is scoped by `owner_id`** — a manager can never see, view, or delete
another's reports (enforced in SQL). A reports message is parsed into **save / list / view / delete**;
only **delete** is destructive. The delete path (`agent/nodes/reports_cmd.py`): parse `{name | client |
today | all}` → **resolve** owner-scoped targets → hit a LangGraph **`interrupt()`** that returns a
blast-radius summary ("⚠ This will permanently delete **N** reports…") and **pauses** (nothing mutated;
the checkpointer persists the pending action) → resume only on an **explicit affirmative** ("confirm" /
"yes") → delete and write an **audit log** entry. A delete with **no qualifier matches nothing** and asks
the user to be specific — a vague request can never propose a mass deletion. `interrupt()` (vs an ad-hoc
"are you sure?" string) makes the pause durable graph state, resumable on the same thread, and testable as
a state transition.

**See it:** as `manager_a` (who seeds three reports), *"Delete all reports mentioning Acme"* → a one-report
summary → `cancel` (safe), then repeat → `confirm`. `manager_b`'s Acme report is untouched — ownership
scoping proven.

### Requirement 4 — Continuous Improvement (the Learning Loop)
**Ask:** (a) per-user format prefs (Manager A likes tables, Manager B bullets); (b) system-level learning
from past interactions.

**4.1 User-level (per-manager memory).** `memory/profiles.py` stores a single **compact free-form
preference description** per `user_id` (not fixed fields) — so *any* preference is captured (layout,
length, currency, metrics to always include). It is injected verbatim into the report prompt alongside the
persona, so Manager A and Manager B get different formats for the *same* question. Standing preferences
are **learned by natural language**: the guard detects "from now on use tables", and `update_prefs`
**merges** it into the existing description with a cheap LLM call (a deterministic append fallback means a
preference is never lost) and persists **synchronously** — so it takes effect on the next question and
survives restarts. One-off asides apply to a single turn.

**4.2 System-level (automatic learning loop).** `memory/feedback.py`. Every completed analysis turn is
captured automatically (question, final SQL, final report, **masked rows**) and run through a three-stage
gate in the **background** — **no 👍/👎, no manual trigger, no human review:**
1. **Deterministic metrics** — executed successfully, non-empty, ≤`LEARNING_MAX_ATTEMPTS` (2)
   self-corrections, no PII incident.
2. **Novelty / dedup** — the question must be novel (max cosine similarity to the bucket below
   `LEARNING_DEDUP_SIMILARITY`=0.97; set deliberately high because in-domain retail questions share a high
   baseline similarity). This also short-circuits the expensive judge for cost control.
3. **LLM-as-judge** — must clear **both** `LEARNING_FAITHFULNESS_BAR` (4/5, grounded in the masked rows)
   **and** `LEARNING_INTENT_BAR` (4/5), so a faithful-but-incomplete answer is never learned.

Approved candidates become `source="learned"` Trios, embedded and retrievable immediately and on next
start. Promotions are **reversible by id** (delete the `learned_*.json`). The decision logic is a pure
function (`decide`) so it is unit-tested without LLM quota. **Why this is the right loop:** it improves
behavior with **no retraining**, gated by objective metrics + faithfulness + intent, and every improvement
is **auditable** (you can see which Trio shaped a later answer).

**See it:** ask a novel, well-formed question; the background gate runs after the answer. Because seed
Trios already cover the common shapes, a near-duplicate is correctly *rejected* by the dedup stage — the
loop is conservative by design.

### Requirement 5 — Resilience & Graceful Error Handling
Covered in depth in [§5](#5-error-handling--fallback-strategies): bounded self-correction, tenacity
retries + a shared circuit breaker over both Gemini and BigQuery, transient/permanent classification, the
dry-run cost guard, and a CLI that never crashes. **See it:** the README's resilience demo seeds a
first-attempt error; `/trace` shows attempt 1 failing and attempt 2 succeeding within budget.

### Requirement 6 — Quality Assurance
**Ask:** how do you evaluate before deployment, and verify that reports answer the user's intent?

**Solution — a test pyramid plus an offline LLM-driven eval harness, all runnable locally.**

- **Unit + component tests** (`tests/`, **140 tests** across 15 files): PII masking, the SQL validator,
  ownership scoping + transactional delete, the oversight interrupt, routing, contextualization,
  clarification, decompose/synthesize, self-correction (fake flaky client), retrieval, preference
  handling, and the learning gate's pure `decide`. A dedicated test asserts **no raw PII reaches a trace**.
- **Offline eval harness** (`src/assistant/eval/`, `make eval`): an **11-case golden set**
  (`tests/eval/golden_set.json`) spanning customer behavior, product performance, time-based, comparative,
  and DB-structure questions, a multi-turn follow-up, a compound question, and **adversarial** cases (PII
  bait, prompt injection, off-topic, small talk). Each case is scored on:
  - **Execution success** — did a valid query run and return rows? (objective)
  - **Result correctness** — for cases with a `reference_sql`, run the authoritative query and check its
    numeric aggregates are reproduced within tolerance in the agent's rows (`eval/correctness.py`).
  - **Intent satisfaction** — LLM-as-judge against the case rubric (0–5).
  - **Faithfulness** — LLM-as-judge **shown the actual masked rows + the SQL**, so a fabricated figure is
    caught, not just internal inconsistency. With no rows it falls back to consistency and **cannot award
    a 5**.
  - **Safety** — adversarial cases assert PII absent / injection refused (objective pass/fail).
- **Release thresholds** (gated, exit 0/1): execution success **≥ 95%**, safety **= 100%**, mean intent
  **≥ 4/5**. The objective `reference_sql` cross-check keeps the judge honest; a **disagreement** (judge
  says good, aggregates differ) is surfaced as the most valuable failure but **reported, not auto-failed**
  — a mismatch can be a legitimate definitional difference (revenue with vs without a status filter).

**Verifying intent specifically** leans on the intent-satisfaction + faithfulness judges cross-checked
against objective correctness, so subjective scoring never stands alone on something measurable.

### Requirement 7 — Observability
**Ask:** know when/why the agent fails; agent-level metrics + deep-dive debugging of message
correspondence.

**Solution — three complementary signals, all keyed by a per-turn `run_id`** (`src/assistant/
observability/`):

1. **Per-turn run trace** (`tracing.py` → `traces/<run_id>.json`, shown by `/trace`) — the deep-dive
   artifact. An ordered, per-node timeline with latencies: the rewritten question, intent, retrieved Trio
   ids, every SQL attempt + error, masked-row count, and outcome. This is the "message correspondence":
   for any complaint, open the trace and see exactly what happened. A per-node **field whitelist**
   (`summarize_delta`) means **only trace-safe fields are recorded — `raw_rows` and `masked_rows` are both
   omitted**, so observability never re-introduces the PII that masking removed (asserted by a unit test).
2. **Agent-level metrics** (`metrics.py`, shown by `/metrics`) — a session accumulator derived **entirely
   from the trace** (so metrics and the trace can never disagree). Implemented counters: requests by
   intent; success / degraded / clarified / rejected (and an **analysis success rate** over completed
   analyses only); self-correction rate + avg attempts/query; empty-result rate; cold-retrieval rate; PII
   cells masked; `pii_leak_prevented`; and average latency/turn.
3. **Structured JSON logs** (`logging.py` → `logs/agent.jsonl`) — one JSON object per record (`level`,
   `logger`, `event`) stamped with `run_id`/`user_id`/`thread_id` from the active turn, so logs are
   greppable locally and queryable in Cloud Logging.

**LangSmith** is optional: set `LANGSMITH_API_KEY` and LangGraph auto-traces each node as a span; without
it the local trace file is the fallback (graceful degradation of observability itself).

> **Honest prototype scope.** The production metric catalogue (Cloud Monitoring + LangSmith) is broader
> than what the prototype computes. **Token / byte / dollar cost telemetry, p50/p95 and per-node latency
> percentiles, and
> dependency-error/breaker-open counters are production targets (Cloud Monitoring + LangSmith) and are
> *not* implemented in the local prototype's `/metrics`.** The prototype implements the run-correlated
> trace, the JSON logs, and the counter set listed above — enough to *demonstrate* the production story on
> a laptop. We call this out rather than imply full coverage.

### Requirement 8 — Agility (Persona Management)
**Ask:** the CEO changes the report *tone* weekly; non-developers must update instructions without
redeployment.

**Solution.** Tone/voice is **config, not code** (`persona/loader.py`, `data/personas/*.yaml`). A persona
has `tone`, `audience`, `style_rules`, `guardrails`, and a `version`. The loader caches per file and
**hot-reloads on mtime change**, so editing the YAML changes the agent's tone on the **next turn — no
restart, no redeploy** (the literal requirement). A malformed edit keeps the last-known-good persona, so a
bad save never takes the agent down. The active persona (`DEFAULT_PERSONA`, default `concise_exec`; an
alternate `data_storyteller` ships) is composed into the report prompt **on top of** the user's per-manager
preferences — org voice and per-manager format compose cleanly and update independently. In production the
config store is GCS/Firestore with a thin admin surface and versioned rollback.

**See it:** edit `data/personas/concise_exec.yaml` (e.g. change the tone line) and ask the next question —
the report's voice changes with no restart.

---

## 8. Extensibility (new capabilities & data sources)

Extensibility is designed in, not bolted on. The architecture is a set of **seams** — stable construction
points behind which an implementation can change or multiply — wired through one typed state object and one
config surface. Adding a capability or a source means implementing a seam and registering it; it never
requires editing the graph core, the safety layer, or unrelated nodes.

| Seam | Where (prototype) | What it lets you add |
|---|---|---|
| Data / warehouse access | `bigquery/runner.py` (`BigQueryRunner`: `execute_query` / `dry_run` / `get_table_schema`) | A new dataset or warehouse |
| Schema + business annotations | `agent/nodes/schema.py` (`build_schema_context`, `_TABLE_NOTES`, `_BUSINESS_RULES`) | New tables/metric conventions exposed to the model |
| Golden Bucket retrieval | `golden/store.py` (`add`/`search`) + `golden/retriever.py` (`retrieve`/`add_trio`) | A different vector store; per-source Trio namespaces |
| LLM / embeddings provider | `llm/client.py` (`get_chat_model`) + `llm/embeddings.py` | A different model/provider (Vertex, OpenRouter, Ollama) |
| Operational stores | `reports/store.py`, `memory/profiles.py` (SQLite) | Swap to Postgres; new persisted entities |
| Persona / config | `persona/loader.py` | New personas/instruction sets |
| Observability sinks | `observability/` | New trace/metric backends (Cloud, LangSmith, OTel) |

> **Honest note on the seams.** In the prototype these are **concrete classes and factories**
> (`from_settings` / `create`), not formally declared `typing.Protocol`/ABC interfaces — each external
> dependency is simply constructed in exactly one place, which is what makes the prototype→production swap
> mechanical. Promoting the load-bearing seams (data source, retriever) to explicit Protocols is a small,
> natural production-hardening step.

**Worked example — adding an `inventory` dataset:** point a `BigQueryRunner` at the new `dataset_id`;
register its schema + business annotations **including its PII columns** (so masking applies
automatically); extend the SQL allow-list; seed a few Trios; optionally add a glossary note. Nothing in
the agent core changes — masking, cost guards, retrieval, and observability all apply because they live on
the shared seams. **What makes this safe:** safety is **centralized on the shared edges**, so a new table
or capability inherits PII masking and read-only validation by construction — you cannot add a source that
bypasses masking by accident.

---

## 9. Limitations & honesty

What is deliberately simplified in the prototype, stated plainly:

- **Backing stores are local.** In-process NumPy vector store (not Vertex Vector Search), SQLite (not
  Cloud SQL), `InMemorySaver` checkpointer (not Postgres), local files (not Cloud Logging/Monitoring/Trace).
  The graph and trust boundaries are identical; only the stores differ ([§2 mapping](architecture.md#6-prototype--production-mapping)).
- **Eval set is small by design** — 11 cases (~8–12 was the target), a single judge model. It demonstrates
  the full methodology (objective checks + grounded LLM-judge + thresholds), not production-scale coverage.
- **The golden eval set does not contain a clarification case or a destructive-op case.** Those behaviors
  *are* tested — but in the **component/unit layer** (`tests/unit/test_reports_nodes.py`,
  `test_reports_store.py`, `test_input_guard.py`, `test_contextualize_safety.py`), not in the offline
  golden set.
- **The eval harness scores execution, correctness, intent, faithfulness, and safety.** It does **not**
  score preference-adherence or per-case cost/latency budgets; preference behavior is covered by unit
  tests, and cost/latency gating is a planned regression guard.
- **Cost telemetry is not in the local `/metrics`** (see [Requirement 7](#requirement-7--observability)) —
  it is a production (Cloud Monitoring / LangSmith) target.
- **Free-tier rate limits** pace live use; the two-tier model routing, caching, bounded iterations, and
  backoff mitigate this. Enabling pay-as-you-go billing removes the caps for an unconstrained demo.
- **`first_name`/`last_name` are visible by default** so reports can name top customers — a deliberate,
  configurable policy choice (`PII_MASK_COLUMNS`), not an oversight.

These are scope decisions appropriate to a 6–12h prototype whose remit is to demonstrate, on a laptop,
that the production design is sound — not to ship production infrastructure.

---

*See also: [architecture.md](architecture.md) (diagrams + service rationale) · [README](../README.md)
(setup, run, demo).*
