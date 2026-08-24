# NebulonMind

Memory layer for AI assistants, built on top of **NebulonDB**.

Every piece of information that NebulonMind decides to consider shares a common structure called a **Memory Object**, and is stored across three representations: *truth* (traditional DB), *meaning* (vector DB), and *relationships* (graph DB).

---

## 1. Architecture Overview

### 1.1 Memory Object

```
Memory
│
├── identity
│   ├── memory_id
│   ├── user_id
│   └── status            ← MemoryStatus: active | archived (single source)
│
├── content
│   ├── text
│   ├── summary
│   └── structured_data
│
├── classification
│   ├── memory_type
│   ├── category
│   └── source
│
├── importance
│   ├── score
│   ├── confidence
│   └── priority
│
├── lifecycle
│   ├── created_at
│   ├── updated_at
│   ├── last_accessed
│   ├── expires_at
│   └── retention_policy
│
├── semantic
│   └── embedding
│
└── relationships
    ├── entities
    └── relations
```

Example:

```json
{
  "memory_id": "mem_001",
  "user_id": "user_001",

  "content": {
    "text": "My name is Sathya",
    "summary": "User's name is Sathya"
  },

  "classification": {
    "memory_type": "semantic",
    "category": "identity",
    "source": "conversation"
  },

  "importance": {
    "score": 0.99,
    "confidence": 1.0
  },

  "lifecycle": {
    "retention_policy": "permanent"
  }
}
```

### 1.2 Memory Types

Start with a small number:

| Type | Meaning | Example |
|------|---------|---------|
| `WORKING` | Current conversation / context | "Continue the database architecture we discussed." |
| `SHORT_TERM` | Useful temporarily | User is currently debugging HNSW. |
| `LONG_TERM` | Stable user information | Name, age, occupation, skills, preferences |
| `EPISODIC` | Events | User created NebulonMind; User completed Phase 1 |
| `SEMANTIC` | Generalized facts | User works with Python; User builds AI systems |
| `KNOWLEDGE` | External information | Python docs, research papers, technical documentation |

> **MemoryType ≠ RetentionPolicy.** `MemoryType` answers *what the memory is*
> (see table above); `RetentionPolicy` answers *how long to keep it*
> (`temporary` / `session` / `permanent`). They are independent axes: "My
> name is Sathya" can be `semantic` + `permanent`, while "I am debugging
> HNSW today" is `working` + `session`. The Step 2 classifier assigns both
> separately (`classifier.memory_type_for` / `retention_for`).

### 1.3 Storage Mapping

Don't make every database store everything:

```
              Memory Object
                   │
          ┌────────┼────────┐
          │        │        │
          ▼        ▼        ▼
      Traditional Vector   Graph
         DB       DB       DB
```

| Store | Role |
|-------|------|
| Traditional DB | **Truth** — memory_id, user_id, type, category, content, importance, confidence, timestamps, retention, status |
| Vector DB | **Meaning** — memory_id, embedding |
| Graph DB | **Relationships** — entities and relations (e.g. `Sathya --HAS_SKILL--> Python`) |

> Traditional DB = truth, Vector DB = meaning, Graph DB = relationships.
> This separation keeps NebulonMind much easier to maintain.

### 1.4 Memory Repository

One interface above the databases:

```
NebulonMind
     │
     ▼
MemoryRepository
     │
     ├── TraditionalStore
     ├── VectorStore
     └── GraphStore
```

```python
class MemoryRepository:

    def create(self, memory):
        ...

    def get(self, memory_id):
        ...

    def update(self, memory_id, data):
        ...

    def delete(self, memory_id):
        ...

    def search(self, query):
        ...

    def relate(self, memory_id, entity):
        ...
```

The AI interacts with `MemoryRepository` only — never directly with HNSW/Graph/SQL.

### 1.5 Phase 1 Result

At the end of Phase 1:

```python
memory = Memory(
    text="My name is Sathya",
    memory_type="long_term",
    category="identity",
    importance=0.99
)

mind.store(memory)
```

And NebulonMind automatically performs the write across:

```
Memory
  │
  ├── Traditional DB
  │
  ├── Vector DB
  │
  └── Graph DB
```

Phase 1 does **not** yet decide whether something should be remembered.

---

## 2. NebulonDB Mapping

NebulonDB already bundles **vector + graph in one engine** (`NebulonOrbit`: Nova hnswlib + Mesh graph + doc store). Therefore only **2 corpora** are needed, not 3:

| Design (spec) | NebulonDB mapping |
|---|---|
| Traditional DB = truth | COSMOS corpus `mind_truth` — one doc per memory, `memory_id` as `_id`, segment = `user_{user_id}` |
| Vector DB = meaning | ORBIT corpus `mind_semantic` — embedding per memory, metadata carries `memory_id` |
| Graph DB = relationships | Same ORBIT corpus's Mesh — entity nodes (label = entity name), edges like `HAS_SKILL`, plus `MEMORY_OF` edges linking entities ↔ memories |

**Why one ORBIT corpus for vector + graph:** it keeps vector+graph transactions consistent (single `save()`, one WAL) and avoids cross-corpus drift. Truth stays separate because COSMOS is a pure document store with no vector/graph coupling.

### 2.1 Module layout

```
nmd_host/
├── api/
│   ├── client.py         # NebulonDBClient: REST client for the NebulonDB API (port 6969)
│   └── server.py         # create_app(): the NebulonMind service itself — /api/NebulonMind/* (port 9696)
├── serve.py              # uvicorn entrypoint: python -m nmd_host.serve (port 9696)
├── core/
│   ├── config.py         # NebulonDBConfig: .env + default endpoint config
│   ├── models.py         # Memory, MemoryContent, Classification, Importance, Lifecycle (pydantic)
│   ├── repository.py     # MemoryRepository protocol + InMemoryRepository + NebulonMindRepository (API-backed)
│   └── mind.py           # NebulonMind facade: store(memory), recall(query), relate(...)
└── stores/
    ├── truth_store.py    # API-backed wrapper → COSMOS corpus mind_truth
    ├── vector_store.py   # API-backed wrapper → ORBIT corpus mind_semantic (embed server-side)
    └── graph_store.py    # API-backed wrapper → same ORBIT corpus Mesh
```

### 2.2 Store semantics

- **TruthStore (COSMOS):** full Memory JSON minus embedding — `memory_id`, `user_id`, type, category, content, importance, timestamps, retention, status. `update` = overwrite doc; `delete` = `delete_data(segment, memory_id)`.
- **VectorStore (ORBIT):** `insert_vec(vector, metadata={memory_id, text, summary, type, category})`. Use `vec.tolist()` (WAL JSON safety) and call `initialize_or_flush()` after each write batch (WAL checkpoint — no premature `.ndb` segment files below the flush threshold).
- **GraphStore (ORBIT Mesh):** entities resolved by label; edges
  `(entity → entity, relation)` and `(memory_id → entity, "HAS_ENTITY")`.
  `add_edge` is idempotent, so re-storing a memory never duplicates edges.
  Every `Relationship` carries `source_memory_id` provenance — stamped by
  `Memory.stamp_provenance()` at store time and persisted in the truth doc —
  so "why do we believe Sathya knows Python?" is always answerable via the
  supporting memory. Deleting a memory removes only its own mesh node and
  incident edges; entity nodes shared with other memories survive.

### 2.3 `NebulonMind.store(memory)` orchestration

```
store(memory):
  1. generate memory_id (uuid / mem_%05d)
  2. truth: insert doc (committed source of truth)
  3. semantic: embed text → insert vector (metadata.memory_id)
  4. graph: for each relationship {entity, relation}: resolve_node(entity label) + add_relation
  5. initialize_or_flush() on the ORBIT corpus
  → on failure at any step: compensate (delete the partial writes), return error
```

Write order: **truth → vector → graph**. Delete order: **graph → vector → truth**. All keys are `memory_id`-based, so retries are naturally idempotent.

### 2.4 `recall(query, top_k)` read path

`NebulonMind.recall()` is the candidate provider; the Step 3
`MemoryRetriever` owns the single retrieval pipeline:

1. Embed query → search on `mind_semantic` → top-k `memory_id`s (k × candidate multiplier)
2. Hydrate each from TruthStore → full `Memory` objects
3. Optional graph expansion: `get_neighbors(memory_id)` → related entities → extra candidates (`expand=true`)
4. Step 3 pipeline over the candidates: retention filter → rank (semantic/importance/confidence/recency) → deduplicate → top-k

Ranking runs before deduplication so a duplicate group never survives as
two entries and only the best-ranked representative is kept.

### 2.5 Multi-tenant isolation

Segment per user (`user_{user_id}`) in both corpora — Cosmos already filters by segment on every read. One `memory_id` space per user.

---

## 3. Scope

### 3.1 Phase 1 includes

- `Memory` model + validation
- `MemoryRepository` interface + in-memory fake (enables tests without a DB)
- TruthStore / VectorStore / GraphStore implementations
- `NebulonMind.store()` + rollback, `recall()`, `relate()`
- REST API routes (open internal service — no authentication; persisted solely through NebulonDB)
- End-to-end test: store → close → fresh engine → recall → assert no duplicates/loss

### 3.2 Phase 1 explicitly excludes

- "Should I remember this?" decision
- Importance decay / expiry sweeps (lifecycle fields are stored only)
- Consolidation / forgetting

---

## 4. Implementation Order

1. `Memory` model + validation
2. `MemoryRepository` protocol + in-memory fake
3. TruthStore
4. VectorStore
5. GraphStore (each with a small smoke test)
6. `NebulonMind.store()` + rollback, `recall()`, `relate()`
7. End-to-end test on the validated pattern: store → close → fresh engine → recall → no duplicates/loss
8. REST API routes (FastAPI, no auth)

---

## 5. Open Design Decisions

1. One ORBIT corpus for vector + graph (recommended) vs two separate corpora?
2. Segment per user vs per memory type?
3. Include the REST API layer in Phase 1, or only the Python `NebulonMind` library?
4. Module location: sibling `nmd_host/` package next to `ndb_host/`, or inside `ndb_host/`? (Resolved: sibling, API-backed)

---

## 6. Phase 2 — Memory Intelligence Layer (Step 2)

Phase 1 stores whatever it is handed. Phase 2 decides **what to remember**:
it turns a raw conversation into fully-classified, scored memory candidates
and persists only the ones worth keeping.

```
                 Conversation
                      │
                      ▼
          ┌──────────────────────┐
          │ MemoryDecisionEngine │   pure: no storage, no network
          └──────────┬───────────┘
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
     Remember?              Don't remember (logged, dropped)
          │
          ▼
   MemoryCandidate           ← validated before it is ever stored
          │
     ┌────┼────┬──────────┐
     ▼    ▼    ▼          ▼
   Type  Score Entity  Relations
          │
          ▼
  converter.candidate_to_memory()
          │
          ▼
         Memory (Step 1 model)
          │
          ▼
       NebulonMind
          │
  ┌───────┼────────┐
  ▼       ▼        ▼
Truth  Vector    Graph
```

### 6.1 New module layout

```
nmd_host/intelligence/
├── schemas.py       # Conversation, Turn, MemoryCandidate, MemoryDecision, MemoryCategory
├── extractor.py     # MemoryExtractor protocol (rules and LLM share this contract)
├── rules.py         # RuleBasedExtractor — deterministic, works without any LLM
├── engine.py        # MemoryDecisionEngine — conversation → validated decisions
├── classifier.py    # category → MemoryType / retention; keyword-signal fallback
├── scoring.py       # importance / confidence / priority
├── entities.py      # rule-based NER: capitalized tokens + tech whitelist
├── relations.py     # subject → relation → object triples for the graph store
├── converter.py     # MemoryCandidate → Step 1 Memory
├── bridge.py        # MemoryIntelligence facade → any repository / NebulonMind
├── validation.py    # drops malformed/hallucinated decisions before storage
├── providers.py     # LLMProvider protocol + OpenAI/Anthropic/Gemini/Qwen/Ollama
└── llm_extractor.py # LLMExtractor — the LLM as another MemoryExtractor
```

### 6.2 Decision schema

| Model | Role |
|---|---|
| `Turn` / `Conversation` | input: a (multi-turn) conversation, `role` + `content` |
| `MemoryCandidate` | one thing that could be remembered: text, summary, `category`, `importance` (0–1), `confidence` (0–1), `entities`, `relationships` |
| `MemoryDecision` | the verdict: `candidate` + `should_remember` + `reason` |

`MemoryCategory`: `identity`, `preference`, `skill`, `goal`, `project`,
`fact`, `event`, `task`, `opinion`. Categories map onto Step 1 types:
identity/preference/skill/goal → `long_term`, project/fact → `semantic`,
event → `episodic`, task → `working`, opinion → `short_term`.

Relationship triples carry `source_memory_id` provenance once stored
(`NebulonMind.store()` stamps it), so every graph claim points back to the
memory that supports it.

### 6.3 Extractors — same contract, different brains

```python
class MemoryExtractor(Protocol):
    def extract(self, conversation: Conversation) -> List[MemoryDecision]: ...
```

* `RuleBasedExtractor` — pattern table, sentence matching, subject resolution
  ("My name is Sathya" then "I work with Python" → `Sathya -HAS_SKILL-> Python`).
  Pure and deterministic; the prototype runs with zero LLM dependency.
* `LLMExtractor` — prompts an `LLMProvider` for the same JSON decision schema,
  then passes output through `validation` (2.13) before the engine accepts it.
  The LLM is an implementation of the extractor — it never controls storage.

### 6.4 Usage

```python
from nmd_host import MemoryIntelligence, RuleBasedExtractor, LLMExtractor, Conversation
from nmd_host.core.mind import NebulonMind

mind = NebulonMind(user_id="user_001")

# deterministic pipeline — no LLM needed
intelligence = MemoryIntelligence(mind, user_id="user_001")          # RuleBasedExtractor by default
memories = intelligence.process(Conversation.from_user_message(
    "My name is Sathya. I work with Python. I like hiking."
))
# 3 Memory objects stored across truth + vector + graph

# same contract, LLM brain (NMD_LLM_PROVIDER=openai|anthropic|gemini|qwen|ollama)
from nmd_host import provider_from_env
intelligence = MemoryIntelligence(mind, user_id="user_001",
                                  extractor=LLMExtractor(provider_from_env()))
memories = intelligence.process_text("Sathya is building NebulonMind.")
```

The bridge accepts any repository-like store (`NebulonMind`,
`NebulonMindRepository`, `InMemoryRepository`), so the whole pipeline is
testable without a database.

### 6.5 Phase 2 includes / excludes

**Includes:** decision schema, extractor contract, rule-based extractor,
decision engine, classification, importance/confidence scoring, entity +
relationship extraction, candidate → `Memory` conversion, persistence bridge,
LLM provider adapters, output validation, offline + end-to-end tests.

**Excludes (Phase 3+):** importance decay, consolidation/forgetting sweeps,
cross-conversation subject reconciliation (the subject is resolved per
conversation only), automatic conflict detection between old and new memories.

---

## 7. Phase 3 — Memory Lifecycle & Retrieval Intelligence (Step 3)

Phase 2 decides **what becomes memory**; Phase 3 manages memory **over time**
and decides **what is relevant now**: retention, expiration, deduplication,
ranking, retrieval, consolidation, forgetting and context building. It works
with already-created `Memory` objects and consumes Step 1 repositories — it
never imports `ndb_host` and never creates another database.

```
MemoryLifecycleManager
  ├── RetentionEvaluator   policy + expiry (PERMANENT / TEMPORARY / SESSION)
  ├── DuplicateDetector    layered: memory_id → normalized text → category+content → semantic
  ├── MemoryRanker         semantic × importance × confidence × recency (weights sum to 1)
  ├── MemoryRetriever      candidates × 3 → retention filter → rank → dedupe → top-k
  ├── MemoryConsolidator   conservative KEEP/UPDATE/MERGE/IGNORE recommendations
  ├── ForgettingManager    cleanup of expired temporaries / ended sessions only
  └── MemoryContextBuilder bounded, provenance-carrying LLM context
```

### 7.1 Data flow (Step 2 → Step 3 → persistence)

```
Conversation
     │
     ▼
MemoryDecisionEngine (Step 2)            ← "this sentence looks like a preference"
     │
     ▼
MemoryDecision → candidate_to_memory()
     │
     ▼
MemoryLifecycleManager.ingest()          ← Step 3 gate before persistence
     ├── lifecycle / retention validation   (STORE / INVALID / EXPIRED)
     ├── duplicate check                     (DUPLICATE with the existing memory)
     └── consolidation check                 (advisory, never auto-merge)
     │
     ▼
NebulonMind.store() → truth + vector + graph
```

### 7.2 Key behaviours

* **Retention** — `PERMANENT` never auto-expires; `TEMPORARY` expires at
  `expires_at`; `SESSION` lives until the session ends. A `TEMPORARY`
  memory without `expires_at` is retained (safe) but flagged as an invalid
  configuration — never an invented expiry.
* **Deduplication** — exact `memory_id` → normalized text equality → same
  category + highly similar content → semantic similarity through the
  existing vector store. `DuplicateResult` states the outcome explicitly.
* **Ranking** — four configurable, validated weights
  (`NMD_RANKING_*_WEIGHT`, sum = 1.0). Recency is a bounded exponential
  decay (never `1/age`), independent of importance. Retention acts as a
  hard gate before ranking.
* **Retrieval** — fetches `top_k × candidate_multiplier` vector candidates
  (never `top_k` raw), filters expired, ranks the full candidate set, then
  deduplicates (keeping the best-ranked memory of each duplicate group)
  and returns `top_k`. User isolation is preserved by the per-user
  segment plus a `user_id` guard.
* **Consolidation** — purely advisory. Similar memories → `MERGE` candidate;
  conflicting facts ("I like Python" / "I don't like Python anymore") →
  `IGNORE`; version drift ("Python 3.10" → "Python 3.13") → `UPDATE`
  preserving historical provenance. Nothing is auto-merged or deleted.
* **Forgetting** — only policy-eligible memories (expired temporaries,
  ended sessions) are deleted, always through
  `ForgettingManager → MemoryRepository → NebulonMind.delete()`.
* **Context** — bounded by item count and characters; ordered by relevance,
  then importance, then recency; every entry carries provenance
  (`memory_id`, type, category, importance, confidence, source).

### 7.3 Module layout

```
nmd_host/lifecycle/
├── lifecycle.py      # MemoryState, is_expired, get_state, age / inactivity
├── retention.py      # RetentionEvaluator (retain / delete / valid-config)
├── deduplication.py  # DuplicateDetector + DuplicateResult + text similarity
├── ranking.py        # RankingConfig (env-driven) + MemoryRanker + recency decay
├── retrieval.py      # RetrievalConfig + MemoryRetriever (pipeline)
├── consolidation.py  # MemoryConsolidator + ConsolidationDecision
├── forgetting.py     # ForgettingManager (find_expired/forget/cleanup)
├── context.py        # ContextConfig + MemoryContextBuilder
├── config.py         # LifecycleConfig + env helpers (NMD_* variables)
└── manager.py        # MemoryLifecycleManager + build_lifecycle_manager
```

### 7.4 Usage

```python
from nmd_host import build_lifecycle_manager
from nmd_host.core.mind import NebulonMind
from nmd_host.core.repository import NebulonMindRepository

mind = NebulonMind(user_id="user_001")
manager = build_lifecycle_manager(NebulonMindRepository(mind), searcher=mind)

results = manager.retrieve("What do I know about Sathya's Python work?", top_k=5)
context = manager.build_context(results)          # bounded LLM context
verdict = manager.ingest(new_memory)              # STORE / DUPLICATE / EXPIRED / INVALID
manager.cleanup(user_id="user_001")               # forget eligible memories
```

The bridge (`MemoryIntelligence.process`) still stores directly; wiring
`manager.ingest()` before `NebulonMind.store()` gives the full
conversation → decision → lifecycle gate → store pipeline. The `GET
/api/NebulonMind/search` endpoint now delegates to `MemoryLifecycleManager`
(a single retrieval system), and `POST /api/NebulonMind/memory/context`
exposes bounded context building.

### 7.5 Phase 3 includes / excludes

**Includes:** lifecycle state evaluation, retention, expiration,
deduplication (incl. semantic via existing vector infra), configurable
ranking, recency/importance/confidence scoring, candidate retrieval, user
isolation, conservative consolidation, automatic forgetting, context
building with provenance, offline + E2E tests.

**Excludes (Phase 4+):** general-purpose autonomous agents (Step 6 ships a
bounded, memory-only tool-calling runtime: remember/recall, fixed max turns,
no arbitrary tool/SDK surface), RL/fine-tuning/training, new
vector/graph/memory databases, LLM-based consolidation, personality
inference.

## 8. Agent Integration

NebulonMind is designed to be plugged into other agents as a memory layer.
The complete guide is in **`docs/AGENT_INTEGRATION.md`**; the short version:

- **Pattern 1 — Delegate chat**: `POST /api/NebulonMind/agent/chat` runs
  NebulonMind's own memory-aware tool loop.
- **Pattern 2 — Memory backend**: write with
  `POST /api/NebulonMind/memory?gate=true` (idempotent, auto-de-dups —
  returns HTTP 200 with the canonical memory on a duplicate) and read with
  `GET /api/NebulonMind/search` / `POST /api/NebulonMind/memory/context`,
  injecting the bounded context into your own prompt.
- **Pattern 3 — Hybrid**: chat for turn-taking plus
  `POST /api/NebulonMind/intelligence/process` to extract durable facts.

Every route scopes to a registered username via the `user_id` query
parameter (create one with
`POST /api/NebulonMind/user/create_user`). Optional hardening knobs:

- `NMD_API_AUTH_TOKEN=<secret>` — require `Authorization: Bearer <secret>`
  on every call (health/metrics/OpenAPI stay public).
- `NMD_AGENT_DURABLE_SESSIONS=true` — persist agent sessions in NebulonDB
  so multi-turn sessions survive a restart.
- `NMD_BACKGROUND_PERSIST_STATE=true` — persist background job run history
  across restarts.
- With `NMD_LLM_EXTRACTOR=true`, extraction falls back to the deterministic
  rules engine whenever the LLM fails *or returns an empty payload*.
  
## 9. Recent Fixes and Improvements

- **API key stored in `.env`**: The LLM API key is now written to `.env` and never
  stored in the configuration file (`nebulonmind.cfg`). The `.env` value is the
  authoritative source; the cfg entry is ignored. This ensures the key remains
  masked in the UI and survives server restarts.
- **Typing indicator persists**: The "AI is typing" animation now remains visible
  until the LLM answer fully arrives, instead of disappearing after a fixed
  timeout. This gives users confidence that the request is still processing.
- **Chat history survives refresh**: On page reload, the console attempts to
  restore the most recent conversation from the server (via `/chats`). If the
  active chat exists only in localStorage, it is reconciled back to the server
  so it cannot be lost. Previously, a refresh could wipe the conversation.
- **No-cache headers on console assets**: The server now serves console JS and
  CSS with `Cache-Control: no-cache`, ensuring browsers always revalidate and
  pick up the latest JavaScript on refresh (no more stale main.js).
- **Dead code cleanup**: Removed unused variables and CSS rules that served no
  purpose: `isTypingAnimation`, `isProcessingCommand`, `messageCount` (JS), and
  `.ascii-text`, `.cursor-blink`, `.code-block` CSS rules. The codebase is
  leaner and inspections are cleaner.
- **Agent sessions clarified**: The `/sessions` command lists in-process agent
  sessions created via `POST /agent/session`; the default console chat is
  stateless and does not create sessions, which is intentional for durability
  across refreshes.
- **Shared constants centralized (`nmd_host/utils/constants.py`)**: All branding
  and operational defaults now live in exactly one place — the pyfiglet banner
  (`NEBULONMIND_BANNER`, font `smslant`), app name, brand tagline, default
  username, TUI usage text, host/port defaults (`API_HOST_DEFAULT`,
  `API_PORT_DEFAULT`, `NEBULONDB_API_HOST_DEFAULT`, `NEBULONDB_API_PORT_DEFAULT`),
  graceful-shutdown default and the background-job cron schedules. Every former
  hardcoded duplicate (`tui/app.py` banner, `tui/commands.py`, `tui/serve.py`,
  `api/server.py`, `api/routes/dashboard.py`, `core/config.py`) now imports them,
  so a value is defined once and used everywhere.
- **Server startup experience**: `nebulonmind start` now prints the NEBULONMIND
  banner, clears every stale `__pycache__` directory under the project (skipping
  `.git`, `.venv`, `node_modules`, …) so freshly edited sources are always used,
  and shows *"Getting the server ready for you..."* before booting uvicorn.
- **Scheduled auto-delete (memory forgetting)**: Expired memories are now swept
  automatically. A third background job `auto_delete_expired` runs daily (cron
  configurable) and calls `ForgettingManager.cleanup()` — only policy-eligible
  memories are deleted (expired `TEMPORARY`, ended `SESSION`); `PERMANENT` and
  `ARCHIVED` are never touched. Controlled from `nebulonmind.cfg`:

  ```ini
  [lifecycle]
  nmd_temporary_ttl_seconds = 2592000        # TTL stamped on temporary memories
  nmd_lifecycle_auto_cleanup = true          # master switch for the sweep
  nmd_lifecycle_auto_cleanup_cron = 0 3 * * *  # when the sweep runs (daily 03:00)
  ```

  With the flag off the job reports `skipped` instead of deleting. Job status is
  visible via `GET /api/NebulonMind/background/status`.

## 10. Roadmap (planned, not yet implemented)

- **Voice chat (LLM + speech)**: speak-to-chat with spoken answers — mic → STT →
  the existing `/agent/chat` pipeline → TTS. Two usage styles are planned:
  *chatbot mode* (push-to-talk, simple REST round-trip) and *call-center mode*
  (continuous listening, streaming STT/TTS over WebSocket, barge-in). Providers
  would be API-key based (OpenAI Whisper API / NVIDIA Riva / Deepgram for STT;
  Azure Speech / ElevenLabs for TTS) configured through a future `[voice]`
  section in `nebulonmind.cfg`, with keys kept in `.env`. Voice is a front-end
  only — no changes to the memory core.
