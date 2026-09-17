# NebulonMind Monitor (`nmd_monitor`) — Plan & Build Log

> Own LangSmith-equivalent for NebulonMind. Local-first observability +
> evaluation + monitoring. No third-party service, no new database:
> everything persists in **NebulonDB** (same as memories), everything is
> scoped by `user_id`, everything fails open (a monitor failure never
> fails a chat).

## ✅ COMPLETED — P1 Feedback + Console Monitor (v0.1.5)

### ✅ P1 Feedback Store & API
- **Model**: `MonitorFeedback` with `trace_id`, `user_id`, `score`, `tag`, `comment`, `by`, `created_at_ms`
- **Store**: Added `save_feedback()` and `get_feedback()` to both `InMemoryMonitorStore` and `NebulonDBMonitorStore`
- **API**: 
  - `POST /monitor/feedback` — submit feedback with score (1-5), tag, comment
  - `GET /monitor/feedback?trace_id=...&limit=50&offset=0` — retrieve user's feedback

### ✅ P1 Console Monitor Tab
- **Commands**:
  - `/monitor` — list recent traces (last 10, with ✅/❌ status)
  - `/monitor <trace_id>` — show trace details with spans, tools, latency
  - `/feedback <trace_id> <score> [tag] [comment]` — submit feedback (1-5 stars)
- **UI**: Terminal-style display with trace previews, span trees, and feedback forms

### ✅ Configuration Rename
- **Short keys**: `NDB_API_*` and `NMD_API_*` (e.g., `NDB_API_HOST`, `NMD_API_PORT`)
- **Legacy support**: Old `NEBULONDB_API_*` and `NEBULONDMIND_API_*` still work
- **Updated**: Config, TUI, dashboard, docs, imports

### 🔧 Next Phase (P2)
- Evaluation runner & metrics
- Advanced UI for trace analysis
- Automated feedback-based learning

---

## 1. What LangSmith does (and what we clone)

| LangSmith concept | `nmd_monitor` equivalent | Status |
|---|---|---|
| Project | `project` field on every trace (default `default`) | P0 done |
| Trace (one request) | one `POST /agent/chat` -> one `MonitorTrace` (`trace_id`) | P0 done |
| Run / Span (one step) | `MonitorSpan{name, kind=llm\|retrieval\|tool\|grounding, latency_ms, ok}` | P0 done |
| Thread (multi-turn) | `thread_id` = `session_id or conversation_id or trace_id` | P0 done |
| Metadata (model, tokens) | `provider, model, tokens, turns, request_id` | P0 done |
| Feedback (human score) | `POST /monitor/feedback {trace_id, score, tag, comment}` | P1 planned |
| Dataset (saved tests) | `mind_datasets` corpus + `POST /monitor/datasets/from-trace` | P2 planned |
| Experiment (scored run) | `POST /monitor/experiments/run` over `EvaluationRunner` | P2 planned |
| Dashboards / online evals | `GET /monitor/stats` + console Monitor tab | P3 planned |

What we deliberately **do not** clone in V1: Prompt Hub versioning,
PagerDuty/webhook alerting, OpenTelemetry export, per-token cost tables,
multi-project RBAC.

## 2. Architecture

```
User text -> /agent/chat (server.py)
    |
    |- grounding retrieve (manager.retrieve) --\
    |- AgentRuntime.chat()  (engine.py)         } spans[]
    |     |- llm.turn (providers.complete)     |
    |     |- recall / remember (tools.py) -----/
    |
    └─► recorder.capture(...)  (monitor/recorder.py, fire-and-forget)
            └─► TraceStore.save() -> NebulonDB COSMOS corpus `mind_traces`
                    segment `user_{user_id}`, doc_type `monitor_trace`
                                     |
GET /monitor/traces  <───────────────┘
GET /monitor/trace/{id}
GET /monitor/stats
```

### 2.1 Package layout (`nmd_host/monitor/`)

| File | Role |
|---|---|
| `models.py` | `MonitorSpan`, `MonitorTrace`, `MonitorFeedback`, `MonitorStats` (pydantic, dependency-free). `from_execution_trace()` converts the existing `ExecutionTrace` so agent code never changes shape. |
| `recorder.py` | `MonitorRecorder`: `enabled`, `sample_rate`, `redact`, `max_body_chars`, `project`. `should_capture()`, `capture()` (sampling + truncation + fail-open), `from_chat()` factory. No threads in P0 — direct save, wrapped in try/except at call site. |
| `store.py` | `MonitorTraceStore`: `save/trace/get/list/stats` over any object with `bundle.repository` semantics. `InMemoryMonitorStore` (dict, used by tests + `InMemoryServiceProvider` path) and `NebulonDBMonitorStore` (real path via `NebulonDBClient`, corpus `mind_traces`). |
| `decorators.py` | `@monitored(name, kind)` + `span(name, kind)` context manager — the `@traceable` equivalent for custom code. Records latency, ok/error without changing return values. |
| `evaluators.py` | Heuristic online-eval helpers over stored traces: `answered_with_zero_recall`, `tool_error_rate`, `empty_recall_rate`, `fallback_used_rate` + `summarize_online()`. Offline scoring stays in `evaluation/metrics.py`. |
| `config.py` | `MonitorConfig.from_env()` (`NMD_MONITOR_*`). Pure function, no import cycle with `core/config.py`. |

### 2.2 Storage (why NebulonDB, not a new DB)

* Follows `stores/chat_history_store.py`: one JSON doc per trace in `text`, `doc_type="monitor_trace"`, segment `user_{user_id}`.
* No new ops: NebulonDB on `:6969` is already required. No SQLite/Postgres to back up.
* Isolation free: per-user segments = one user can never list another's traces.
* Trade-off accepted: trace list is `list-then-filter` in Python (same as chats); fine for P0 volumes (<10k traces/user). Paginate with `limit/offset`.

### 2.3 API (`nmd_host/api/routes/monitor.py`, prefix `/api/NebulonMind/monitor`)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/monitor/traces?user_id=&q=&tool=&ok=&limit=&offset=` | List traces (newest first), filter by text/tool/outcome |
| GET | `/monitor/trace/{trace_id}?user_id=` | Full trace incl. spans |
| GET | `/monitor/stats?user_id=&days=` | Counts, err%, avg latency/tokens, recall/remember usage |
| POST | `/monitor/feedback` | (P1) attach human score to a trace |
| POST | `/monitor/datasets/from-trace` | (P2) promote a bad trace to regression dataset |
| POST | `/monitor/experiments/run` | (P2) run `EvaluationRunner` over a dataset, persist report |

Envelope: `{success, message, data}` like every other route. `user_id` required (invalid -> 400, unknown -> 403 via `_bundle`, same as chats).

### 2.4 Config (`nebulonmd.cfg [monitor]` + env)

```ini
[monitor]
nmd_monitor_enabled = true
nmd_monitor_project = default
nmd_monitor_sample_rate = 1.0
nmd_monitor_redact = true
nmd_monitor_max_body_chars = 2000
nmd_monitor_retention_days = 90
```

Env override: `NMD_MONITOR_ENABLED`, `NMD_MONITOR_PROJECT`,
`NMD_MONITOR_SAMPLE_RATE`, `NMD_MONITOR_RED consistent with `NMD_*`
convention. No secrets (nothing to redact from cfg).

### 2.5 Redaction & safety

* `redact=true` (default): `input_text/answer` truncated to `max_body_chars`; span inputs/outputs truncated the same way; raw memory bodies never stored (counts only — mirrors `ExecutionTrace` contract).
* Sampling: `random() < sample_rate`; `/health`, `/metrics`, `/monitor/*` never traced (no recursion).
* Fail-open: recorder + store + route wiring all wrapped; `logger.warning` only. Chat path in `server.py` guards with `try/except` so monitor downtime = missing trace, not 500.

## 3. Build log

### P0 (this change) — persist + list + stats

- [x] `nmd_host/monitor/` package (models, recorder, store, decorators, evaluators, config)
- [x] `nmd_host/api/routes/monitor.py` router (traces, trace detail, stats)
- [x] `server.py`: `Monitor` tag + `include_router` + fire-and-forget capture in `agent_chat`
- [x] `[monitor]` section in `nebulonmd.cfg` + `ServiceConfig(monitor_*)` + `NMDConfig._load_monitor()` + settings group
- [x] `unittest/test_monitor.py` (offline, no DB): models, recorder sampling/redaction, store CRUD, decorators, API routes via `InMemoryServiceProvider`
- [x] Docs: this file + README docs-table row + `FILE_STRUCTURE.md` row

### P1 (next) — feedback + console tab

* `mind_feedback` corpus + `POST /monitor/feedback` + thumbs up/down in web console + annotation filter (`ok=false`, `score<3`).
* Console `Monitor` tab: trace list -> span tree with latency bars, thread view by `session_id`.

### P2 (after) — datasets + experiments

* `mind_datasets` / `mind_experiments` corpora, `from-trace` promotion, `experiments/run` wiring `EvaluationRunner` + `evaluation/metrics.summarize()`, side-by-side v1-vs-v2 view.

### P3 (later) — retention + online evals UI

* Scheduler cron (`agents/scheduler.py`) deleting traces older than `retention_days`; `stats` charts; LLM-as-judge evaluator via `provider_from_env()`.

## 4. Verification

```bash
# offline (no NebulonDB needed)
uv run pytest unittest/test_monitor.py unittest/test_observability.py -q

# manual (needs NebulonDB + LLM key)
curl -X POST "http://localhost:9696/api/NebulonMind/agent/chat?user_id=nmd_user_01" \
  -H "Content-Type: application/json" -d '{"text":"my name is Sathya"}'
curl "http://localhost:9696/api/NebulonMind/monitor/traces?user_id=nmd_user_01&limit=5"
curl "http://localhost:9696/api/NebulonMind/monitor/stats?user_id=nmd_user_01"
```

## 5. File reference (P0)

```
docs/MONITOR_PLAN.md            # this file
nmd_host/monitor/__init__.py
nmd_host/monitor/models.py
nmd_host/monitor/recorder.py
nmd_host/monitor/store.py
nmd_host/monitor/decorators.py
nmd_host/monitor/evaluators.py
nmd_host/monitor/config.py
nmd_host/api/routes/monitor.py
unittest/test_monitor.py
nebulonmd.cfg ([monitor])
```
