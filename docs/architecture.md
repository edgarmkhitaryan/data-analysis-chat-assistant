# Architecture & Diagrams

This document is **Deliverable 1** — the architecture diagrams (building blocks, services, and flow)
with a short rationale for every named framework, service, and data store. The deep technical
explanation lives in **[HLD.md](HLD.md)**; the runnable prototype is described in the
**[README](../README.md)**.

The system is designed **production-first** and then realized as a faithful local prototype: the
LangGraph control flow and the trust boundaries are identical in both; only the backing stores differ
(managed GCP services in production, in-process / SQLite / local-file equivalents in the prototype).
That mapping is explicit in [§6](#6-prototype--production-mapping).

---

## 1. Production system architecture

The headline diagram. A thin client talks to a stateful **LangGraph** agent on **Cloud Run**, grounded by
two knowledge sources — the **BigQuery** warehouse (facts) and the **Golden Bucket** of analyst Trios
(interpretation) — and wrapped in cross-cutting layers for safety, memory/persona, resilience, and
observability. Every external box is a named, justified service ([§5](#5-why-each-service-the-rationale)).

```mermaid
flowchart TB
    subgraph Client["Client / Channel"]
        CLI["CLI (prototype)"]
        Chat["Slack / Web chat (prod)"]
    end

    subgraph Edge["Edge & Identity"]
        IAP["IAP / Identity Provider<br/>maps user → manager_id"]
    end

    subgraph Agent["Agent Orchestrator — LangGraph on Cloud Run"]
        direction TB
        Ctx["Contextualize<br/>follow-up → standalone"]
        Clarify["Clarify<br/>ask when ambiguous"]
        Guard["Input Guard<br/>injection filter + intent"]
        Route{"Route by intent"}
        Decompose["Decompose<br/>compound → sub-questions"]
        Synth["Synthesize<br/>merge sub-reports"]
        subgraph Analysis["Analysis pipeline (per sub-question)"]
            direction TB
            Retrieve["Retrieve Golden Trios"]
            Schema["Schema provider (cached)"]
            GenSQL["Generate SQL"]
            ValSQL["Validate SQL<br/>read-only · allow-list · LIMIT"]
            Exec["Execute on BigQuery<br/>dry-run cost guard"]
            Correct["Self-correct loop<br/>(bounded)"]
            Mask["PII Masking"]
            Report["Synthesize Report<br/>persona + prefs"]
        end
        subgraph Reports["Report-management path"]
            ParseCmd["Parse command"]
            Resolve["Resolve targets<br/>(owner-scoped)"]
            Confirm["Confirm<br/>human-in-the-loop interrupt"]
            Mutate["Execute delete / save<br/>(audited)"]
        end
        Prefs["Update preference<br/>standing → profile"]
    end

    subgraph LLM["Vertex AI"]
        Gemini["Gemini (chat)"]
        Embed["gemini-embedding-001"]
    end

    subgraph Data["Data & Knowledge"]
        BQ[("BigQuery<br/>thelook_ecommerce — read-only")]
        GCS[("GCS — Golden Bucket<br/>Trios data lake")]
        VEC[("Vertex AI Vector Search<br/>Trio embeddings")]
    end

    subgraph Ops["Operational stores"]
        PG[("Cloud SQL / Postgres<br/>Saved Reports · Profiles · Checkpointer")]
        CFG[("Config store (GCS / Firestore)<br/>Persona / Instructions")]
        SEC[("Secret Manager")]
    end

    subgraph Obs["Observability"]
        LOG["Cloud Logging"]
        MON["Cloud Monitoring (alerts)"]
        TRACE["Cloud Trace"]
        LS["LangSmith (LLM traces + evals)"]
    end

    subgraph Learn["Learning pipeline (asynchronous)"]
        PS["Pub/Sub"]
        FN["Cloud Functions / Workflows<br/>auto quality gate + curation"]
    end

    CLI --> IAP
    Chat --> IAP
    IAP --> Ctx
    Ctx -->|ambiguous| Clarify
    Ctx -->|resolved / first turn| Guard --> Route
    Route -->|analysis| Decompose --> Retrieve
    Route -->|manage reports| ParseCmd
    Route -->|preference| Prefs
    Prefs -.->|also asks a question| Decompose

    Retrieve --> Schema --> GenSQL --> ValSQL --> Exec --> BQ
    Exec -->|error / empty| Correct --> GenSQL
    Exec -->|rows| Mask --> Report --> Synth

    Retrieve --> VEC
    Retrieve --> Embed
    Ctx --> Gemini
    GenSQL --> Gemini
    Report --> Gemini
    Synth --> Gemini

    ParseCmd --> Resolve --> Confirm --> Mutate
    Resolve --> PG
    Mutate --> PG
    Prefs --> PG
    Confirm -.->|awaits confirm| CLI

    CFG -. persona / tone .-> Report
    GCS --> FN --> VEC
    Synth -. candidate event .-> PS --> FN
    Agent --> SEC
    Agent --> LOG
    Agent --> MON
    Agent --> TRACE
    Agent --> LS
```

**How to read it:** solid arrows are the request/response path within a turn; dotted arrows are
asynchronous or human-in-the-loop edges (the confirmation pause, and the *fully automatic* learning loop
that runs after the answer is returned). The learning loop is deliberately **off the request path** so it
never adds user-facing latency.

---

## 2. The agent graph (LangGraph topology — as built)

This is the **actual** control flow compiled by `build_graph()` in
[`src/assistant/agent/graph.py`](../src/assistant/agent/graph.py). The agent is two compiled graphs: a
**checkpointed outer graph** (conversation, routing, oversight) and a **stateless inner analysis
pipeline** that `run_compound` invokes once per sub-question (a single question is just the degenerate
"one sub-question" case). Routing is implemented as conditional-edge functions, and the safety/PII
boundaries are wired as topology — not as prompt instructions.

```mermaid
flowchart TD
    START((start)) --> ctx["contextualize<br/>(injection filter → rewrite follow-up)"]
    ctx -->|ambiguous| clarify["clarify (ask one question)"] --> E1((end))
    ctx -->|resolved / first turn| guard["guard_input<br/>(injection filter + intent)"]
    guard -->|rejected| reject["respond_reject"] --> E2((end))
    guard -->|ok| load["load_context<br/>(persona + prefs)"]
    load --> route{"route by intent"}

    route -->|update_preference| upd["update_prefs<br/>(merge + persist)"]
    upd -->|preference only| E3((end))
    upd -->|also asks a question| decompose

    route -->|analysis| decompose["decompose<br/>(compound?)"]
    decompose --> runc["run_compound<br/>(N× analysis pipeline)"]
    runc --> synth["synthesize<br/>(merge; partial-failure tolerant)"]
    synth --> E4((end))

    route -->|manage_reports| parse["parse_report_command"]
    parse -->|save| save["save_report (audited)"] --> E5((end))
    parse -->|list| list["list_reports"] --> E5
    parse -->|view| view["view_report (owner-scoped)"] --> E5
    parse -->|delete| resolve["resolve_targets<br/>(owner-scoped)"]
    resolve -->|none matched| none["respond_none<br/>(ask to be specific)"] --> E6((end))
    resolve -->|matched| confirm{{"confirm_delete<br/>interrupt() → CLI confirm"}}
    confirm --> E6

    subgraph Pipeline["Analysis pipeline — compiled subgraph, no checkpointer (reused per sub-question)"]
        direction TB
        p0((start)) --> retr["retrieve_golden"] --> sch["get_schema"] --> gen["generate_sql"]
        gen --> val["validate_sql"]
        val -->|valid| exe["execute_sql"]
        val -->|invalid · attempts left| sc["self_correct"]
        exe -->|rows / accepted empty| mask["mask_pii"]
        exe -->|error / empty · attempts left| sc
        sc --> gen
        val -->|attempts exhausted| deg["degrade"]
        exe -->|attempts exhausted| deg
        mask --> rep["synthesize_report"] --> pEnd((end))
        deg --> pEnd
    end

    runc -.runs.-> Pipeline
```

Key properties visible here (full discussion in [HLD §4](HLD.md#4-data-flow-between-components)):

- **`mask_pii` is the only edge** from `execute_sql` to `synthesize_report` — the report LLM physically
  cannot receive unmasked rows.
- **The self-correction loop is a graph loop**, bounded by `MAX_SQL_ATTEMPTS` (default 3), not prompt
  recursion — so every attempt is an inspectable, metered state transition.
- **`confirm_delete` is a LangGraph `interrupt()`** — the pause is durable graph state, resumable on the
  same thread.
- **The learning loop is *not* a node.** It runs after the turn responds, on a background thread in the
  CLI (Pub/Sub + Cloud Functions in production). See [§4](#4-the-automatic-learning-loop).

---

## 3. Data flow — the analysis happy path (with self-correction)

A single conversational turn, end to end, including a self-correction round-trip.

```mermaid
sequenceDiagram
    autonumber
    participant U as Manager (CLI)
    participant G as LangGraph agent
    participant GB as Golden Bucket (retriever)
    participant L as Gemini
    participant V as SQL validator (sqlglot)
    participant BQ as BigQuery (read-only)
    participant T as Tracer / Metrics

    U->>G: "Which categories drove the most revenue last quarter?"
    G->>G: contextualize (rewrite vs history) + injection filter
    G->>L: classify intent (cheap model) → analysis
    G->>GB: embed question → top-k similar Trios
    GB-->>G: exemplar (question → SQL → report) Trios
    G->>L: generate_sql (schema + Trios + question)
    L-->>G: candidate SQL
    G->>V: validate (read-only, allow-list, LIMIT)
    V-->>G: ok (LIMIT injected)
    G->>BQ: dry-run cost estimate, then execute
    BQ-->>G: ERROR "Unrecognized name: revenue"
    G->>G: self_correct (feed exact error back) — attempt 2
    G->>L: regenerate SQL with error context
    L-->>G: corrected SQL
    G->>BQ: execute (attempt 2)
    BQ-->>G: rows
    G->>G: mask_pii (deterministic, pre-LLM)
    G->>L: synthesize_report (masked rows + Trios + persona + prefs)
    L-->>G: analyst report
    G->>T: persist trace + update metrics
    G-->>U: report (+ run_id, /trace, /metrics)
    Note over G,T: After responding, a background thread runs the<br/>automatic learning gate on this turn (see §4).
```

---

## 4. The automatic learning loop

Every completed analysis turn becomes a **candidate Trio** and is run through a fully automatic,
three-stage gate — **no user feedback, no manual trigger, no human in the loop**. It runs in the
background so it never adds latency, and stages are ordered cheapest-and-most-decisive-first so the
expensive LLM judge only ever sees novel, already-successful turns.

```mermaid
flowchart LR
    turn["Completed analysis turn<br/>(question, SQL, report, masked rows)"] --> cap["Capture candidate<br/>(automatic)"]
    cap --> m{"Stage 1 — metrics<br/>executed · non-empty · ≤2 retries · no PII incident"}
    m -->|fail| drop["Discard"]
    m -->|pass| d{"Stage 2 — novelty<br/>max similarity < 0.97?"}
    d -->|duplicate| drop
    d -->|novel| j{"Stage 3 — LLM judge<br/>faithfulness ≥ 4 AND intent ≥ 4"}
    j -->|fail| drop
    j -->|pass| norm["Normalize → Trio (source=learned)"]
    norm --> up["Embed + upsert"]
    up --> idx[("Golden Bucket index")]
    idx -.->|retrieved next time| turn
```

Promotions are **reversible by id** (delete the `learned_*.json`); the index self-heals via a content
fingerprint. Detail and thresholds: [HLD §8, Requirement 4](HLD.md#requirement-4--continuous-improvement-the-learning-loop).

---

## 5. Why each service (the rationale)

Per the brief: every named framework / service / store, with the reason it was chosen.

| Building block | Service (production) | Why this one |
|---|---|---|
| **Orchestration** | **LangGraph** (+ LangChain Core) | The workflow is a graph with a **loop** (self-correction) and a **pause** (confirmation). LangGraph models loops, conditional routing, and **human-in-the-loop `interrupt()`** as first-class concepts, and its **checkpointer** gives durable, resumable conversation state. The brief also prefers LangGraph/LangChain. |
| **Compute** | **Cloud Run** | Stateless, autoscaling, scale-to-zero container host; durable state lives in Cloud SQL, so the agent tier scales horizontally and cheaply. |
| **LLM** | **Gemini** via **Vertex AI** | Recommended by the brief; strong SQL/reasoning; the **same key/identity provides embeddings**. Vertex gives IAM-based auth (no raw API keys), higher quotas, and data-residency controls. Two-tier routing (a cheap model for low-stakes structured calls, the main model for SQL/report generation) cuts cost and spreads rate-limit load. |
| **Embeddings** | **`gemini-embedding-001`** | One provider for chat + embeddings; 3072-dim vectors; co-located with the model and warehouse. |
| **Warehouse** | **BigQuery** (read-only) | Mandated; the public `thelook_ecommerce` dataset is rich enough for every expected capability. Read-only is enforced at the **IAM** layer (prod) *and* the **SQL-validation** layer (always). A **dry-run byte estimate + max-bytes cap** prevents runaway cost. |
| **Golden Bucket lake** | **GCS** | Cheap, durable object store — the literal "data lake" of analyst Trios; the source of truth that the vector index is built from. |
| **Vector index** | **Vertex AI Vector Search** | Managed ANN at scale, co-located with the model/embeddings. The prototype's in-process cosine store sits behind the same retriever interface, so the swap is mechanical. |
| **Operational stores** | **Cloud SQL (Postgres)** | Transactional deletes (oversight), per-user ownership, and durable conversation state (it also backs the **LangGraph Postgres checkpointer**) in one managed, transactional store. |
| **Persona / config** | **GCS / Firestore** | Tone/instructions are **config, not code**, so a non-developer (the CEO) edits them weekly with versioning + rollback and **no redeploy**. |
| **Secrets** | **Secret Manager** | No secrets in images, code, or env files in production. |
| **Observability** | **Cloud Logging / Monitoring / Trace** + **LangSmith** | The same signals we'd alert on, plus a hosted LLM-trace/eval surface. The prototype emits the *same* run-correlated signals to file-based sinks. |
| **Learning pipeline** | **Pub/Sub + Cloud Functions / Workflows** | Decouples candidate capture from the (idempotent, retryable) quality-gate + curation job, so promotion scales independently and never touches the request path. |

**RAG over fine-tuning** is the other defining choice: analyst knowledge changes continuously, so
retrieval lets us change behavior by *adding a Trio* — no retraining, and full traceability of *why* an
answer was shaped a certain way. Reasoning in depth: [HLD §3](HLD.md#3-reasoning-for-the-chosen-cloud-llm-and-frameworks).

---

## 6. Prototype ↔ production mapping

The prototype keeps the **same graph and the same trust boundaries**; only the backing stores change.
This is itself the extensibility story — each swap happens behind one factory (`from_settings` / `create`).

| Concern | Prototype (this repo) | Production |
|---|---|---|
| Orchestration | LangGraph + LangChain Core (in-process) | Same, on Cloud Run |
| LLM | Gemini via `langchain-google-genai` (API key) | Gemini via Vertex AI (IAM) |
| Embeddings | `gemini-embedding-001` | Same (Vertex AI) |
| Vector store | In-process NumPy cosine (cached to disk) | Vertex AI Vector Search |
| Warehouse | BigQuery via ADC | BigQuery via workload identity, read-only IAM |
| Golden Bucket lake | `data/golden_trios/*.json` | GCS bucket |
| Saved Reports · profiles · checkpointer | SQLite (`data/app.db`) + `InMemorySaver` | Cloud SQL (Postgres) + Postgres checkpointer |
| Persona config | `data/personas/*.yaml` (hot-reload on edit) | GCS / Firestore |
| Secrets | `.env` | Secret Manager |
| Observability | JSON logs + per-turn trace files + `/metrics` summary | Cloud Logging / Monitoring / Trace + LangSmith |
| Learning pipeline | Inline capture + background-thread promotion | Pub/Sub + Cloud Functions / Workflows (same gate) |

---

## 7. The PII trust boundary

The single most important safety property, drawn explicitly. PII is removed **before** any row reaches
the LLM or the user — it is *impossible-by-construction*, not discouraged-by-prompt. Observability never
re-introduces it: **no row data — raw *or* masked — is persisted to traces, logs, or metrics**; only
counts, ids, SQL text, and sizes are recorded (the per-node whitelist in `summarize_delta` drops both
`raw_rows` and `masked_rows`).

```mermaid
flowchart LR
    BQ[("BigQuery<br/>(may return email, address, geo)")] -->|raw_rows<br/>in memory only| MASK["mask_pii<br/>(deterministic: columns + regex)"]
    MASK -->|masked_rows| REP["synthesize_report (LLM)"]
    REP --> OUT["output guard<br/>(re-scan report text)"]
    OUT --> USER["Manager"]

    MASK -. counts / ids only .-> TR[("Traces / Logs / Metrics")]
    REP -. report size only .-> TR

    classDef danger fill:#fde,stroke:#c33;
    classDef safe fill:#dfe,stroke:#3a3;
    class BQ danger;
    class USER,TR safe;
```

The output guard is the *alarm*, not the guarantee: a non-zero `pii_leak_prevented` count means a bug to
fix, even though the user was still protected. Mechanics: [HLD, Requirement 2](HLD.md#requirement-2--safety--pii-masking).
