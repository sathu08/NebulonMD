# NebulonMind — Architecture

NebulonMind is the memory layer for AI assistants, built on top of
**NebulonDB**. Every memory is persisted in three representations — truth
(COSMOS documents), meaning (ORBIT vector embeddings) and relationships
(ORBIT Mesh graph) — and every layer of NebulonMind talks to NebulonDB
exclusively through its REST API. No `ndb_host` Python code is imported
anywhere.

```
┌────────────────────────── NebulonMind (nmd_host) ────────────────────────┐
│                                                                          │
│  Clients:                                                                │
│    Web console (web_dir)          REST clients (curl, agents, apps)      │
│              │                              │                            │
│              ▼                              ▼                            │
│  ┌───────────────────  API service (port 9696)  ─────────────────────┐   │
│  │  routes → per-user ServiceBundle (repository + lifecycle manager) │   │
│  │  middleware: correlation-id / body-limit / rate-limit / metrics   │   │
│  │  error envelope {success, message, data}; /metrics (Prometheus)   │   │
│  └───────────────┬───────────────────────────────────────────────────┘   │
│                  ▼                                                        │
│  AI layer (repository protocol)                                           │
│    intelligence  →  lifecycle manager  →  agent runtime / agents         │
│                  │                                                       │
│                  ▼                                                       │
│        MemoryRepository (protocol)                                       │
│          ├── InMemoryRepository          (tests, no DB)                  │
│          └── NebulonMindRepository  ──►  NebulonMind facade              │
│                                                  │                       │
│                                      TruthStore / VectorStore /          │
│                                      GraphStore                          │
│                                                  │                       │
│                                                  ▼                       │
│                                          NebulonDBClient                 │
│                                      (requests + HTTP Basic)             │
└──────────────────────────────────────┼───────────────────────────────────┘
                                       │  REST (port 6969)
                             ┌─────────▼──────────┐
                             │   NebulonDB server │
                             └────────────────────┘
```

## Layers

The codebase is organised into distinct layers; each layer talks only to the
interface immediately below it.

| Layer | Package | Responsibility |
|---|---|---|
| Web console | `nmd_host/web_dir/` | static single-page console (terminal-chat UI, plus legacy Chat / Evaluation / Background assets). Presentation only — always calls the REST API, never the backend. |
| API service | `nmd_host/api/` | FastAPI service on port 9696. Routes, error envelope, middleware, per-user `ServiceBundle` dependency injection, web-console hosting. |
| AI layer | `nmd_host/agent/`, `nmd_host/agents/`, `nmd_host/intelligence/`, `nmd_host/lifecycle/`, `nmd_host/evaluation/` | All behaviour: deciding what to remember, lifecycle & retrieval, the chat agent, scheduled background agents, evaluation. Talks only to `MemoryRepository`. |
| Repository | `nmd_host/core/repository.py` | The single persistence interface. Two implementations: in-memory (tests) and NebulonDB-backed. |
| Facade + stores | `nmd_host/core/mind.py`, `nmd_host/stores/` | `NebulonMind` coordinates the three stores; each store maps one representation onto a NebulonDB corpus/segment. |
| Backend client | `nmd_host/api/client.py` | Thin HTTP client over the NebulonDB REST API. |

Two rules keep the architecture honest:

* **All memory persistence goes through the REST API.** `nmd_host` never
  imports `ndb_host`; there is no local memory database, no local vector
  store, no SQLite, no Redis. NebulonDB is the only persistent memory store.
  NebulonMind may keep local operational/configuration files — `.env` (config
  endpoint) and `evaluation/reports/` (benchmark output) — but these are not
  memory storage.
* **The AI layer never touches stores directly.** It talks to
  `MemoryRepository` (protocol). NebulonDB-backed persistence is reached
  only via `NebulonMindRepository → NebulonMind → stores → client`.

## Domain model

`nmd_host/core/models.py` (pydantic) defines the memory domain.

`Memory` is the unit of persistence:

* `memory_id` (optional until first store), `user_id`
* `content` — `MemoryContent{text, summary, structured_data}`
* `classification` — `Classification{memory_type, category, source}`
* `provenance` — `Provenance{source_type, created_by, session_id, conversation_id}`
* `importance` — `Importance{score, confidence, priority}`
* `lifecycle` — `Lifecycle{created_at, updated_at, last_accessed, expires_at, retention_policy}`
* `status` — `MemoryStatus{ACTIVE, ARCHIVED}` (the single authoritative status field)
* `entities`, `relationships`

Two enums are deliberately orthogonal:

* `MemoryType` — *what* the memory is (working / short_term / long_term /
  episodic / semantic / knowledge).
* `RetentionPolicy` — *how long* to keep it (temporary / session /
  permanent).

So "My name is Sathya" is `semantic` + `permanent`, while "debugging HNSW
today" is `working` + `session`.

`Relationship{source, target, relation, source_memory_id}` models a directed
entity → entity edge. `source_memory_id` is provenance (which memory's content
supports this claim), stamped automatically by `Memory.stamp_provenance()` and
persisted in the truth document — the graph is a *derived* view that can
always be re-grounded via its supporting memory.

`Memory.to_truth_doc()` / `from_truth_doc()` convert to/from the JSON payload
stored in the truth store.

## Persistence: stores over the NebulonDB REST API

`nmd_host/stores/` maps each representation to a NebulonDB corpus, with one
segment per user (`user_{user_id}`) for multi-tenant isolation.

| Store | Corpus | Engine | What it holds |
|---|---|---|---|
| `TruthStore` | `mind_truth` | cosmos | One document per memory; the full `Memory` JSON lives in the `text` column (the engine only persists `{text, lang, type, created_at}` and assigns its own integer `_id`), so reads scan the segment, parse JSON and match on `memory_id`. |
| `VectorStore` | `mind_semantic` | orbit | One embedding per memory, computed server-side (`is_precomputed=False`); the record's label carries the `memory_id`. |
| `GraphStore` | `mind_semantic` | orbit Mesh | Entities as nodes (auto-resolved by label); each memory is a node linked to its entities via `HAS_ENTITY`; user relationships become directed entity → entity edges. Writes are idempotent. |

Reads for the graph are *derived from the truth store* (the API exposes no
label → node resolution), so `delete_memory` is a no-op: deleting a memory's
vector record removes its mesh node and incident edges. Deletion never
removes shared entity nodes (e.g. `Python` shared by two memories survives
deleting one).

### Write path (`NebulonMind.store`)

1. resolve `memory_id` + timestamps + provenance
2. `TruthStore.create(doc)` — the authoritative record
3. `VectorStore.update(id, text)` — server-side embedding
4. `GraphStore.relate_memory(memory)` — entity edges

On any failure after the truth write, `_compensate()` deletes graph → vector
→ truth (inverse order) and re-raises — partial writes never survive, and a
failed store is never silently half-persisted. Delete order is the inverse:
graph → vector → truth.

### Read path (`NebulonMind.recall`)

`recall()` is the **candidate provider only**: it embeds the query
server-side, fetches top-k × candidate-multiplier `memory_ids`, optionally
adds depth-1 graph expansion (`expand=True`), hydrates each via the truth
store, and hands the candidates to the lifecycle `MemoryRetriever`, which
runs the full post-candidate pipeline (see Lifecycle).

## Repository layer

`nmd_host/core/repository.py` defines `MemoryRepository` (a protocol) — the
only interface the AI layer talks to:

```
create / get / update / delete / list_all / search / relate / close
```

* `InMemoryRepository` — dict-backed fake with a token-overlap search
  scorer; used by tests and the offline evaluation runner. No database.
* `NebulonMindRepository` — API-backed adapter that delegates every
  operation to a `NebulonMind`, so the AI layer stays storage-agnostic.

## Memory intelligence

`nmd_host/intelligence/` decides **what** should become a memory.

```
Conversation ──► MemoryDecisionEngine ──► validated MemoryDecisions
                     │                            │
                extractor                     converter
             (rules | LLM)                        ▼
                     │                      Memory ──► repository / store
                     ▼
        MemoryCandidate (text, category, importance,
                         confidence, entities, relations)
```

| File | Responsibility |
|---|---|
| `schemas.py` | `Conversation`/`Turn` input; `MemoryCandidate`, `MemoryDecision`, `MemoryDecisionList`, `MemoryCategory` |
| `extractor.py` | `MemoryExtractor` protocol — one contract for the rule-based and LLM extractors |
| `rules.py` | `RuleBasedExtractor`: a pattern table (identity, preference, skill, goal, …), sentence splitting, subject resolution, weak-signal rejection |
| `llm_extractor.py` | `LLMExtractor`: fills the same decision schema via any `LLMProvider` (JSON mode); output is sanitized per item |
| `engine.py` | `MemoryDecisionEngine`: extract → drop `should_remember=False` → validate |
| `classifier.py` | category → `MemoryType` / retention mapping + keyword-signal fallback |
| `scoring.py` | category-based importance, pattern-strength confidence, priority |
| `entities.py` | capitalized-token NER + tech whitelist (word-boundary matched) |
| `relations.py` | category → relation map (`HAS_SKILL`, `PREFERS`, `BUILDS`, …) and triple builder |
| `converter.py` | `candidate_to_memory()` — the only place decision output meets the `Memory` model |
| `validation.py` | per-item sanitization: parse, drop empty/out-of-range/invalid, fix entities, dedupe |
| `bridge.py` | `MemoryIntelligence` facade: conversation → decisions → store; accepts a repository or a `NebulonMind` |
| `providers.py` | `LLMProvider` protocol; OpenAI / Anthropic / Gemini / Qwen / Ollama adapters; `provider_from_env()` |

The extraction engine is pure — no storage, no network. Persistence is the
bridge's job. Extraction never requires an LLM: the rule-based extractor is
the default, and the LLM extractor only fills the same schema (its output
still passes the validation layer). All provider SDKs are optional imports.

## Memory lifecycle

`nmd_host/lifecycle/` decides **when** a memory is alive, retrievable, worth
merging or ready to forget.

```
Conversation → intelligence → candidate_to_memory()
      │
      ▼
MemoryLifecycleManager.ingest()   ← lifecycle/retention/duplicate gate
      │                              (STORE / INVALID / EXPIRED / DUPLICATE)
      ▼
repository.create(memory) → truth + vector + graph
      (NebulonMindRepository.create → NebulonMind.store)

MemoryLifecycleManager
  ├── RetentionEvaluator     PERMANENT / TEMPORARY / SESSION expiry
  ├── DuplicateDetector      memory_id → normalized text → category+content → semantic
  ├── MemoryRanker           semantic / importance / confidence / recency blend (weights sum = 1)
  ├── MemoryRetriever        candidates × multiplier → filter → rank → dedupe → top-k
  ├── MemoryConsolidator     conservative KEEP / UPDATE / MERGE / IGNORE recommendations
  ├── ForgettingManager      expired temporaries / ended sessions → repository.delete
  └── MemoryContextBuilder   bounded context with provenance (memory_id, source, …)
```

### The ingest gate

`MemoryLifecycleManager.ingest(memory)` is the single gate before any memory
is stored. In order:

1. invalid retention configuration (TEMPORARY without `expires_at`) → `INVALID`
2. already expired → `EXPIRED`
3. duplicate of an existing memory → `DUPLICATE` (with the existing one)
4. otherwise → `STORE`

### Retention (`RetentionEvaluator`)

* `PERMANENT` — never automatically expires.
* `TEMPORARY` — expires at `expires_at`; without it the configuration is
  flagged invalid (the system never invents an expiry).
* `SESSION` — retained until the session ends.

Nothing is ever deleted merely because it is old. `ARCHIVED` is frozen, not
deleted. (`MemoryState` — active / expired / session / archived — is a
*derived* lifecycle classification computed from `status`,
`retention_policy`, `expires_at` and session state; `status` remains the
only authoritative persisted field.)

### Deduplication (`DuplicateDetector`)

Four layered strategies: exact `memory_id` match → normalized-text equality →
same category + highly similar content (token-level, ≥ 0.90) → semantic
similarity via the existing vector infrastructure (≥ 0.92). Lexical helpers
(`normalize_text`, `text_similarity`, `jaccard_similarity`) live here and are
reused by ranking.

### Ranking (`MemoryRanker`)

One score blending four independent signals, with weights validated to sum
to 1.0 (defaults `0.50 / 0.20 / 0.15 / 0.15`):

```
final = semantic×w_semantic + importance×w_importance
      + confidence×w_confidence + recency×w_recency
```

The semantic signal is a query-to-memory similarity **re-scored over the
vector-selected candidates** — lexical token overlap by default, or an
injected `SemanticScorer`. The vector store's own similarity `score` is used
only to select candidates (`recall()` returns hydrated `Memory` objects, not
scores), so ranking does not depend on that score; it recomputes relevance
itself. This keeps the pipeline storage-agnostic (the in-memory repository
uses the same lexical scorer).

Recency is a bounded exponential decay on `updated_at` (`2^(-age/half_life)`),
so recent memories score near 1 and old ones decay toward 0 but never
collapse. Retention acts as a hard *gate before* ranking, never a ranking
signal.

### Retrieval (`MemoryRetriever`)

The single retrieval pipeline, used by both the API (`/search`,
`/memory/context`) and the agent's `recall` tool:

```
query
  ▼
candidate provider (vector search, over-expanded × multiplier)
  ▼
retention filtering (expired / ended sessions excluded)
  ▼
ranking (full candidate set)
  ▼
deduplication (keeps the best-ranked memory of each duplicate group)
  ▼
top-k memories
```

Ranking runs *before* deduplication so a duplicate group never survives as
two entries and the best-ranked representative is the one kept. A `user_id`
guard preserves per-user isolation on top of the user-scoped segment.

### Consolidation (`MemoryConsolidator`)

Asks whether several memories represent the same *evolving* fact — and then
deliberately does very little. Similar memories are **never** automatically
merged or deleted; the output is a recommendation only:

* conflicting facts ("I like Python" vs "I don't like Python anymore") → `IGNORE`
* version drift ("Python 3.10" → "Python 3.13") → `UPDATE` (recommendation)
* near-identical content + category → `MERGE` (candidate)

Every decision names its `source_memory_ids` so callers can trace the
reasoning to the exact memories involved.

### Forgetting (`ForgettingManager`)

`cleanup()` deletes memories eligible under lifecycle policy (expired
temporaries, ended sessions) — through the repository only
(repository → `NebulonMind` → graph → vector → truth). PERMANENT and
ARCHIVED are never forgotten.

### Context (`MemoryContextBuilder`)

Turns retrieved memories into a bounded, provenance-carrying context for an
LLM: highest-relevance first, then importance, then recency, capped by item
count and character budget. Pure formatting/selection — never calls an LLM,
never mutates the memories.

## Chat agent

`nmd_host/agent/` is the user-facing agent: a stateless tool-calling loop over
the memory system.

### Runtime loop

`AgentRuntime.chat(text, messages)` builds one runtime per request (never
cached — the process stays stateless; anything worth keeping flows to
NebulonDB through the tools):

```
system prompt + transcript (+ tool results) → LLM reply
reply is a tool-call JSON  → execute tool → feed result back
reply is plain text       → final answer
```

The loop is bounded by `max_turns` (default 4). Malformed tool-call JSON is
treated as the final answer; a failing tool never hangs the loop. The
returned `transcript` carries the clean conversation so multi-turn chat
resends history without client-side bookkeeping.

### Memory toolkit

Two tools ship, both backed entirely by the existing intelligence → lifecycle
→ repository pipeline (the agent never touches stores or the client):

* `RememberTool` — conversation → intelligence decisions → lifecycle
  `ingest()` gate → STORE-approved memories persisted (with provenance
  stamped).
* `RecallTool` — lifecycle `MemoryRetriever` → bounded context rendering.

### Sessions

`AgentSessionManager` is an in-process, thread-safe, bounded registry keyed
by `(user_id, session_id)` — **no SQLite or Redis is introduced**, and there
is no session table. Per-user isolation, bounded capacity per user (default
100), lazy TTL expiry (default 3600s). Everything a session holds (transcript,
metadata) is runtime memory; content worth keeping is persisted to NebulonDB
via `remember`.

### Observability (execution trace)

Every run returns an `ExecutionTrace` on the chat response (`AgentChatData.trace`)
so the console can render a Debug/Developer panel:

| Field | Meaning |
|---|---|
| `trace_id` | stable per-run id (uuid hex) |
| `total_ms` / `turns` | wall-clock duration and number of agent turns |
| `llm` | one `LLMSpan` aggregated across turns: model, cumulative latency, approximate tokens (prompt + reply word counts) |
| `tools` | one `ToolSpan` per tool invocation: tool, ok, latency, detail |
| `recall` | optional `RecallSpan` when `recall` was invoked: query, memories found, latency |

Traces are **never persisted** and never carry credentials or raw memory
bodies by default.

## LLM providers

`nmd_host/intelligence/providers.py` defines one `LLMProvider` protocol
(`complete` / `structured` → `LLMResponse{text, provider}`) and swappable,
optional-dependency adapters — OpenAI, Ollama, Qwen, Anthropic, Gemini —
selected via `NMD_LLM_PROVIDER`. Adapters never leak SDK-specific shapes and
raise only the typed `LLM*Error` hierarchy (`LLMTimeoutError`,
`LLMRateLimitError`, `LLMTokenLimitError`, `LLMUnavailableError`,
`LLMInvalidResponseError`), classified from SDK exceptions by type
name/message. `provider_from_env()` wraps the adapter in a bounded,
exponential-backoff `RetryingLLMProvider` on rate limits
(`NMD_LLM_MAX_RETRIES`).

Both the extraction layer and the agent reuse this single provider —
extraction via `NMD_LLM_PROVIDER`/`NMD_LLM_MODEL`, the agent via the same
provider but potentially a different model (`NMD_AGENT_MODEL`).

## Background agents

`nmd_host/agents/` runs scheduled memory work without a user in the loop.
These agents **reuse the existing lifecycle pipeline** — the same
`MemoryConsolidator`, the same ingest gate, the same repository — so they are
a new top layer, not a second consolidation/summarisation system, and the
chat agent never hears about them.

| Component | Responsibility |
|---|---|
| `MemoryAgent` | runs the real `MemoryConsolidator` over a user's memories and reports KEEP / UPDATE / MERGE / IGNORE decisions. Never deletes: sources stay intact; in `consolidate` mode it persists one merged memory per MERGE decision, built from the **newest** source's content (nothing is fabricated) |
| `TaskAgent` | weekly "what did I work on" summary: scans memories in a look-back window (default 7 days), groups by category, renders a summary (template-based by default; an optional LLM only *rewrites* it), and persists it through the ingest gate so it is retrievable like any other memory |
| `BackgroundScheduler` | in-process cron scheduler (5-field expressions, lists + `*/n` steps) running jobs via `asyncio.to_thread`; per-job state is in-memory only — no new persistent store. Start/stop are bound to the API service lifespan |

Default jobs (for `NMD_BACKGROUND_USER`, default `nmd_user_01`): nightly memory
consolidation at `0 2 * * *`, a weekly summary at `0 9 * * 0` (Sundays), and the
auto-delete sweep `auto_delete_expired` — daily by default (`0 3 * * *`).
Schedules live as constants in `nmd_host/utils/constants.py`
(`MEMORY_CONSOLIDATION_CRON_DEFAULT`, `WEEKLY_SUMMARY_CRON_DEFAULT`,
`AUTO_DELETE_CRON_DEFAULT`); the auto-delete cron is overridable via
`nebulonmind.cfg` → `[lifecycle] nmd_lifecycle_auto_cleanup_cron`.
Manual triggers accept any user and are exposed as API endpoints.

The **auto-delete sweep** is the only scheduled job that deletes: gated by
`[lifecycle] nmd_lifecycle_auto_cleanup = true`, it runs
`MemoryLifecycleManager.cleanup()` (→ `ForgettingManager`), removing only
policy-eligible memories (expired `TEMPORARY`, ended `SESSION`) — never
`PERMANENT` or `ARCHIVED`. When disabled it reports a `skipped` result instead.

## Agent evaluation

`nmd_host/evaluation/` benchmarks the **real chat agent** over a bundled
dataset:

* `evaluation/dataset.json` — `memory_test_v1`: 8 questions (4 known-recall
  with seeds, 2 known-no-tool, 2 unknown), each carrying `seed`,
  `expected_memory`, `expected_tool` and `expected_answer`.
* `runner.py` — `EvaluationRunner` drives the agent per question:
  1. seeds the user's memory through the real ingest gate;
  2. measures *retrieval* independently via `manager.retrieve` (did the
     pipeline surface the expected memory?);
  3. runs the agent chat loop, recording chosen tools, final answer, latency
     and approximate tokens. Known items are scored by answer correctness;
     unknown items by hallucination/decline behaviour.
* `metrics.py` — pure, dependency-free metric functions:
  `tool_selection_accuracy`, `retrieval_accuracy`, `answer_correctness`,
  `hallucination_rate`, `latency_stats`, `token_stats`, `summarize`.
* Reports are saved as JSON under `evaluation/reports/` (timestamped).

## API service (port 9696)

`nmd_host/api/server.py` — `create_app(provider=None)` builds the FastAPI
service exposing `/api/NebulonMind/...`. All responses use the standard
envelope `{success, message, data}`; errors use the same envelope with
`success: false` and **no internal exception or credential details** (see
`nmd_host/api/errors.py`; validation errors carry a structured `data.errors`
list; unhandled exceptions are logged and surface as a generic 500).

### Dependency injection

`nmd_host/api/service.py` — routes never construct `NebulonMind` or
repositories themselves; they only talk to the `ServiceBundle` their user's
provider hands out:

* `ServiceBundle{user_id, repository, manager, expand_fn, relate_fn}` — CRUD
  through the repository, recall/context through the lifecycle manager,
  intelligence through the bridge.
* `DefaultServiceProvider` — the production stack: one shared
  `NebulonDBClient` per app, a cached `NebulonMind` per user.
* `InMemoryServiceProvider` — identical interfaces over an
  `InMemoryRepository`; used by offline tests so no NebulonDB is required.

Per-user resources are created on explicit registration and cached for the
process lifetime (`provider.bundle(username)` only resolves registered users;
concurrent requests share the same logical bundle, different users always get
different bundles); `provider.close()` releases everything on shutdown.

### Middleware

`nmd_host/api/middleware.py` — dependency-free operational middleware:
`CorrelationIdMiddleware` (mint/echo `X-Request-ID`, shared with logs and
backend calls), `BodyLimitMiddleware` (413 on oversized bodies),
`RateLimitMiddleware` (token-bucket per client IP, 429 + `Retry-After`),
`MetricsMiddleware` + `MetricsRegistry` (counters/latency rendered in
Prometheus text format at `/metrics`).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/NebulonMind/health`, `/health/live`, `/health/ready` | service/backend health; starts even if NebulonDB is down (reports `backend=down`) |
| GET | `/metrics` | Prometheus metrics |
| POST | `/api/NebulonMind/user/create_user` | register a username → opaque `user_id` (idempotent; 201 created / 201 already-registered) |
| POST | `/api/NebulonMind/memory` | store a memory (201) |
| GET / PUT / DELETE | `/api/NebulonMind/memory/{id}` | fetch / partial update / delete |
| GET | `/api/NebulonMind/search` | lifecycle-ranked semantic recall (`expand=true` → graph expansion) |
| POST | `/api/NebulonMind/memory/context` | bounded, provenance-carrying LLM context |
| POST | `/api/NebulonMind/intelligence/decide` | verdicts only — nothing persisted |
| POST | `/api/NebulonMind/intelligence/process` | verdicts → ingest gate → store (STORE / DUPLICATE / EXPIRED / INVALID) |
| POST | `/api/NebulonMind/memory/{id}/relate` | link memory → entity |
| POST | `/api/NebulonMind/agent/chat` | run the chat agent (stateless loop; optional `session_id`) |
| POST | `/api/NebulonMind/agent/session` | create a session (201) |
| GET | `/api/NebulonMind/agent/session/{id}`, `/api/NebulonMind/agent/sessions` | read one / list active sessions |
| DELETE | `/api/NebulonMind/agent/session/{id}` | close a session (memories remain) |
| GET | `/api/NebulonMind/evaluation/dataset` | bundled benchmark dataset |
| POST | `/api/NebulonMind/evaluation/run` | run the evaluation (503 when no LLM provider is configured) |
| POST | `/api/NebulonMind/background/memory/run` | run the Memory Agent once (mode review/consolidate) |
| POST | `/api/NebulonMind/background/task/run` | run the Task Agent once (days) |
| GET | `/api/NebulonMind/background/status` | scheduler + job state |
| GET | `/api/NebulonMind/llm/status` | configured provider + model (credentials never exposed) |

Every data route takes a `user_id` **query parameter** interpreted as a
**username**; the route resolves it to the opaque per-user `user_id` through
the identity registry (see below) and rejects unregistered usernames with
**403** + `WWW-Authenticate: NebulonMindUser`.

The service also hosts the web console: `nmd_host/web_dir/index.html` (ships
inside the package) is served at the dashboard routes (`/api/NebulonMind/`,
`/index`, `/web`) and a runtime-config view (`GET/PUT
/api/NebulonMind/config`) reads/writes allow-listed values in `.env` (a
restart applies them).

### Identity registry & authentication posture

Every client that touches user data must first **explicitly register** a
username via `POST /api/NebulonMind/user/create_user`. The registry maps the
username to an **opaque `user_id`** (`user_<hex>`), persisted in the
`nmd_Secrets` corpus / `Authentication` segment (NebulonDB) or in memory
(`InMemoryUserRegistry` for tests). All other endpoints treat their `user_id`
query parameter as a *username* and fail with **403** (and
`WWW-Authenticate: NebulonMindUser`) when it is unregistered — there is no
auto-registration anywhere. The mapping is captured into the service bundle
(`provider.bundle(username)`), and a session's `user_id` is that resolved id,
so downstream storage (memories, sessions) is consistently partitioned by the
opaque identity. Reaching the same username again (even after a restart) is
idempotent and returns the same `user_id`.

Beyond that, the NebulonMind API remains an **open internal service** — no
API key or auth middleware on any route. This is not an absence of NebulonDB
authentication: NebulonDB credentials live only inside `NebulonMind` /
`NebulonDBClient`, are never needed by API clients, and are never echoed by
health, metrics, OpenAPI or the config endpoint (asserted by tests).
  User data is *logically partitioned* by the per-user `user_id` segment;
  authorization is intentionally outside the NebulonMind API because the
  service is designed as an internal trusted service (any client may send
  any registered `user_id`, so this is data partitioning, not security
  isolation). No passwords are involved: a username alone identifies the
  client.

## Configuration

Configuration is environment-driven from `.env` (see `nmd_host/core/config.py`).

* **Backend connection** (`NebulonDBConfig`) — `NEBULONDB_API_HOST/PORT`,
  `NEBULONDB_USERNAME/PASSWORD`, `NEBULONDB_API_SCHEME`,
  `NEBULONDB_API_{CONNECT,READ,WRITE}_TIMEOUT` (explicit per-phase timeouts).
* **Service** (`ServiceConfig`) — `NMD_ENV`, `NMD_API_CORS_ORIGINS`,
  `NMD_API_MAX_BODY_BYTES`, `NMD_API_MAX_TOP_K`,
  `NMD_API_MAX_CONTEXT_CHARACTERS`, `NMD_API_RATE_LIMIT_PER_MINUTE`,
  `NMD_API_BACKEND_RETRIES`, `NMD_API_ALLOW_PLAINTEXT_HTTP`,
  `NMD_EXPECTED_BACKEND_VERSION`, `NMD_API_WORKERS`,
  `NMD_API_GRACEFUL_SHUTDOWN_SECONDS`.
* **LLM** — `NMD_LLM_PROVIDER`, `NMD_LLM_API_KEY`, `NMD_LLM_MODEL`,
  `NMD_LLM_TIMEOUT`, `NMD_LLM_MAX_RETRIES`.
* **Agent** — `NMD_AGENT_MODEL`, `NMD_AGENT_MAX_TURNS`, `NMD_AGENT_MAX_RECALL`,
  `NMD_AGENT_TEMPERATURE`, `NMD_AGENT_MAX_SESSIONS`,
  `NMD_AGENT_SESSION_TTL_SECONDS`, `NMD_AGENT_SYSTEM_PROMPT`.
* **Lifecycle** — `NMD_RETRIEVAL_TOP_K`, `NMD_RETRIEVAL_CANDIDATES`,
  `NMD_RANKING_*_WEIGHT`, `NMD_RANKING_RECENCY_HALF_LIFE_DAYS`,
  `NMD_CONTEXT_MAX_ITEMS`, `NMD_CONTEXT_MAX_CHARACTERS`,
  `NMD_TEMPORARY_TTL_SECONDS`, `NMD_LIFECYCLE_AUTO_CLEANUP`,
  `NMD_LIFECYCLE_AUTO_CLEANUP_CRON` (cron for the scheduled auto-delete sweep;
  default `0 3 * * *`).
* **Background agents** — `NMD_BACKGROUND_USER`.

Operational settings are also readable from `nebulonmind.cfg` (INI, loaded into
the environment at startup by `_load_cfg`; secrets stay in `.env`). Shared
defaults — hosts/ports, graceful shutdown, cron schedules, branding — live in
`nmd_host/utils/constants.py` so every module imports the same value instead of
redeclaring it.

`validate_production_config()` is a fail-fast check enforced when
`NMD_ENV=production`: real (non-sample) backend credentials, HTTPS unless
explicitly overridden, and a configured rate limit.

## Testing

### Test layers

1. **Pure-function unit tests** — metrics, cron logic, decision engines.
2. **Repository-level tests** — in-memory + NebulonDB-backed repositories.
3. **API-level tests** — `TestClient` over `InMemoryServiceProvider` (no DB).
4. **E2E tests** — against a live NebulonDB backend (auto-skip when down).

### Offline suite

The offline suite (everything except the live-backend E2E tests) runs and
passes without NebulonDB. Highlights: `test_agent*` (runtime loop, sessions,
system prompt), `test_api*` (endpoints, schemas, reliability), `test_agent`
against a fake LLM, `test_intelligence_*` (rules, engine, bridge, LLM),
`test_lifecycle*`/`test_retention`/`test_ranking`/`test_deduplication`/
`test_consolidation`/`test_context`/`test_forgetting`/`test_retrieval`,
`test_evaluation` (metrics/runner/endpoints over `InMemoryServiceProvider`),
`test_observability` (trace spans on the chat envelope), `test_background_agents`
(cron, scheduler lifecycle, both agents over the real consolidator — asserting
sources are never deleted — and all endpoints).

Only the live-backend E2E tests (`test_*_e2e.py`, `test_stores.py`,
`test_repository.py`) need a running NebulonDB.

## Running the stack

1. Start NebulonDB: `NEBULONDB_HOME=... python nebulondb.py start`
   (binds `0.0.0.0:6969`; port in `nebulondb.cfg` `[server] port`).
2. Start the NebulonMind service — either the managed runner
   (`python nebulonmind.py start|stop|restart [--foreground]`, PID file +
   logs; startup prints the banner, clears `__pycache__`, and shows
   *"Getting the server ready for you..."*) or directly:
   `../NebulonDB/.venv/bin/python -m nmd_host.tui.serve`
   → `http://localhost:9696/api/NebulonMind/`.
3. Tests: `../NebulonDB/.venv/bin/python -m pytest` from this repo root —
   backend-dependent tests skip automatically when NebulonDB is unreachable.

## Design notes / gotchas

- **MemoryType vs RetentionPolicy are orthogonal.** One answers *what* the
  memory is, the other *how long* it is kept. A memory is never classified by
  its retention.
- **Relationship provenance lives in the truth document.** The graph is a
  derived view; `source_memory_id` (stamped by the store layer, persisted in
  the truth doc) is what answers "why do we believe Sathya knows Python?".
  NebulonDB Mesh ignores extra edge fields, so provenance cannot live inside
  the edge itself.
- **Deletion never removes shared entities.** Orbit delete removes only the
  deleted record's mesh node and incident edges; entity nodes used by other
  memories survive.
- The StandardResponse envelope always serializes `data: null`; success of
  `delete_record` is reported via `body["exists"]`.
- COSMOS engines assign their own integer `_id`; the client-supplied
  `memory_id` lives inside the JSON payload, so lookups scan and match on it.
- ORBIT mesh edges are idempotent — re-storing a memory never duplicates
  edges.
- NebulonDB bug fixed upstream (`mesh_store.save()`): stale node/edge rows
  are deleted on save, so removed graph nodes no longer resurrect on reload.
- **Agent traces are never persisted.** The `ExecutionTrace` rides only on
  the `/agent/chat` response; there is no trace store.
- **Background agents never delete arbitrarily.** The consolidation and
  summary agents go through the same ingest gate and never call delete paths —
  the Memory Agent's `consolidate` mode persists a merged memory but leaves its
  sources untouched. The single exception is the dedicated `auto_delete_expired`
  sweep, which deletes only policy-expired memories through `ForgettingManager`
  (the Step 3 forgetting pipeline) and only while
  `nmd_lifecycle_auto_cleanup = true`. Any other background deletion path would
  break this invariant.
- **The Task Agent summary is template-first.** The weekly summary is always
  template-generated; an optional LLM only *re-renders* it (rewrite failures
  fall back to the template). As wired in the service the scheduled/manual
  runs pass no LLM, so the default schedule works without one in the loop.
