# NebulonMind

**NebulonMind** is a memory layer for AI assistants, built on top of
[NebulonDB](https://github.com/sathu08/NebulonDB). It turns raw conversations
into durable, ranked, de-duplicated **memories** and exposes them through a
REST API, a web console, and a memory-aware agent chat endpoint.

Every memory shares one structure — the **Memory Object** — stored across three
representations inside NebulonDB:

| Store | Role |
|---|---|
| Traditional DB (`COSMOS`) | **Truth** — id, type, category, content, importance, timestamps |
| Vector DB (`ORBIT` / Nova) | **Meaning** — semantic embeddings |
| Graph DB (`ORBIT` / Mesh) | **Relationships** — `nebulon_user --HAS_SKILL--> Python` |

On top of storage it adds an intelligence pipeline: decide *what* to remember,
classify and score it, detect duplicates, rank by relevance, expire what is no
longer needed, and build bounded LLM-ready context.

```
Conversation ──► Decision Engine ──► Lifecycle gate ──► NebulonDB
                        │                                   │
                        └────────── recall / rank ◄─────────┘
                                    │
                            Agent chat answer
```

> 🏗️ **Architecture** — for the full internal design (memory model, decision
> engine, retrieval pipeline, agent runtime), see the `docs/` directory in
> this repository.

## 📚 Documentation

| File | Purpose |
|---|---|
| [API_ENDPOINTS.md](docs/API_ENDPOINTS.md) | Complete API reference with all 31 endpoints, request/response examples, and quick start guides |
| [AGENT_INTEGRATION.md](docs/AGENT_INTEGRATION.md) | How to plug NebulonMind into agents — three chat patterns, auth options, caveats |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture: layers, domain model, persistence, agent runtime, LLM providers, background agents |
| [FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md) | Complete project file reference — top-level files, tests/, docs/, nmd_host/, key env vars, quick-start commands |

---

## ⚙️ Prerequisite — NebulonDB must be installed first

NebulonMind stores **everything** in NebulonDB. Without a running NebulonDB
server (default `localhost:6969`), NebulonMind cannot start.

### 1. Install NebulonDB

**Ubuntu / Linux (curl):**
```bash
curl -fsSL https://raw.githubusercontent.com/sathu08/NebulonDB/master/install.sh | bash
```

**Windows (PowerShell, curl):**
```powershell
irm https://raw.githubusercontent.com/sathu08/NebulonDB/master/install.bat -OutFile "$env:TEMP\install.bat"; & "$env:TEMP\install.bat"
```

The installer clones the repo, installs `uv` + Python 3.10, syncs dependencies
and creates a global `nebulondb` command.

### 2. Create an admin user and start it

```bash
nebulondb --create-user     # pick username + password (8+ chars), role super_user
nebulondb start             # serves http://localhost:6969
curl http://localhost:6969/api/NebulonDB/... # API docs at /docs
```

Full details: [NebulonDB README](https://github.com/sathu08/NebulonDB#readme).

---

## 🛠️ Install NebulonMind

**Ubuntu / Linux (curl):**
```bash
curl -fsSL https://raw.githubusercontent.com/sathu08/NebulonMD/main/install.sh | bash
```

**Windows (PowerShell, curl):**
```powershell
irm https://raw.githubusercontent.com/sathu08/NebulonMD/main/install.bat -OutFile "$env:TEMP\install.bat"; & "$env:TEMP\install.bat"
```

What the installer does:

1. Clones this repository into `~/.nebulonmind`
2. Installs `uv` and Python 3.10 if missing
3. Syncs all dependencies into `.venv`
4. Creates a global `nebulonmind` command in `~/.local/bin` (Linux/macOS) or
   the matching location on Windows

Re-running the installer updates an existing clone to the latest `main`.

---

## 🔐 Configuration

Two files live in the install directory (`~/.nebulonmind`):

### `.env` — secrets only

```ini
NMD_LLM_API_KEY=nvapi-...        # key for the provider chosen below
NEBULONDB_USERNAME=...           # NebulonDB admin credentials from step above
NEBULONDB_PASSWORD=...
```

> Provider keys are never written to the cfg file. Get a free NVIDIA NIM key at
> <https://build.nvidia.com> — any OpenAI-compatible provider also works.

### `nebulonmind.cfg` — everything else

| Section | Key | Default | Purpose |
|---|---|---|---|
| `[llm]` | `nmd_llm_provider` | `nvidia` | `nvidia` \| `openai` \| `anthropic` \| `gemini` \| `qwen` \| `ollama` |
| | `nmd_llm_model` | provider default | e.g. `nvidia/nemotron-3.5-lightning-30b-a3b` |
| | `nmd_llm_timeout` / `nmd_llm_max_retries` | `60` / `3` | request tuning |
| `[backend]` | `nebulondb_api_host` / `nebulondb_api_port` | `localhost` / `6969` | where NebulonDB runs |
| `[server]` | `nebulondmind_api_host` / `nebulondmind_api_port` | `0.0.0.0` / `9696` | where Mind listens |
| `[agent]` | `nmd_agent_model` | (uses `nmd_llm_model`) | optional: separate model for agent tool-calling |
| | `nmd_agent_temperature`, `nmd_agent_max_turns`, `nmd_agent_system_prompt` | — | agent-chat behaviour |
| `[ranking]` | `nmd_ranking_*_weight` | sums to 1.0 | retrieval ranking signals |
| `[lifecycle]` | `nmd_temporary_ttl_seconds`, `nmd_lifecycle_auto_cleanup_cron` | 30 days, daily 03:00 | auto-forgetting sweep |
| `[context]` | `nmd_context_max_items` / `nmd_context_max_characters` | `10` / `6000` | LLM context bounds |

---

## ▶️ Running

```bash
nebulonmind start      # background daemon, logs to logs/
nebulonmind restart
nebulonmind stop
```

On start, `__pycache__` directories are automatically cleared to prevent stale bytecode.

Make sure the LLM key reaches the server process (the daemon inherits your
shell environment):

```bash
cd ~/.nebulonmind
set -a; source .env; set +a
nebulonmind start
```

Verify:

```bash
curl http://localhost:9696/api/NebulonMind/health
# {"success":true,...,"backend":"up"}

curl http://localhost:9696/api/NebulonMind/llm/status
# {"data":{"provider":"nvidia","configured":true,...}}
```

---

## 💬 Using NebulonMind

### Web console (recommended)

Open **<http://localhost:9696/api/NebulonMind/dashboard/>** in a browser.
Register/login with your username, then chat — the assistant automatically
remembers facts you share ("my name is Alex", "I work with Python") and
recalls them in later sessions. Chat history survives page refreshes.

**Slash commands in console:**

| Command | Description |
|---|---|
| `/remember "text" [--lang=en] [--type=doc\|semantic\|episodic\|working\|short_term\|long_term\|knowledge]` | Store a memory with optional language and type |
| `/search <query>` | Semantic recall |
| `/context <query>` | Build bounded LLM context |
| `/create <username>` | Register new user |
| `/setup <username>` | Switch to existing user |
| `/whoami` | Show current user |
| `/status` | API + backend + LLM health |
| `/clear` | Clear terminal |
| `/help` | List all commands |

> The terminal CLI/TUI is still **under production** — please use the website
> console for day-to-day usage until it ships.

### REST API — All Endpoints

Every route scopes to a registered username via `?user_id=` (or `username=` for `/user/*`).

| Category | Method | Endpoint | Description |
|----------|--------|----------|-------------|
| **Service** | GET | `/health/live` | Liveness probe (process up) |
| | GET | `/health/ready` | Readiness probe (backend reachable) |
| | GET | `/health` | Combined health (compat) |
| | GET | `/metrics` | Prometheus metrics |
| **User** | POST | `/user/create_user` | Register username (idempotent) |
| | GET | `/user/resolve` | Resolve username → user_id |
| | POST | `/user/setup` | Activate existing username |
| **Memory** | POST | `/memory` | Store memory (with `gate=true` runs lifecycle gate) |
| | GET | `/memory/{id}` | Get memory by ID |
| | PUT | `/memory/{id}` | Update memory |
| | DELETE | `/memory/{id}` | Delete memory |
| | GET | `/search` | Semantic recall (ranked) |
| | POST | `/memory/context` | Build bounded LLM context |
| | POST | `/memory/{id}/relate` | Link memory to entity |
| **Intelligence** | POST | `/intelligence/decide` | Extract decisions (no storage) |
| | POST | `/intelligence/process` | Extract → lifecycle gate → store |
| **Agent** | POST | `/agent/chat` | **Main chat**: tool-calling loop (recall, remember, decide) |
| | POST | `/agent/session` | Create persistent session |
| | GET | `/agent/session/{id}` | Get session |
| | GET | `/agent/sessions` | List sessions |
| | DELETE | `/agent/session/{id}` | Close session |
| **Background** | POST | `/background/memory/run` | Run MemoryAgent (consolidate/review) |
| | POST | `/background/task/run` | Run TaskAgent (weekly summary) |
| | GET | `/background/status` | Scheduler status |
| **Evaluation** | GET | `/evaluation/dataset` | Get bundled test dataset |
| | POST | `/evaluation/run` | Run evaluation suite |
| **LLM** | GET | `/llm/status` | Provider health + model info |

#### Quick Examples

```bash
# Register user
curl -X POST "http://localhost:9696/api/NebulonMind/user/create_user?user_id=nmd001"

# Agent chat (uses recall + remember + decide tools internally)
curl -X POST "http://localhost:9696/api/NebulonMind/agent/chat?user_id=nmd001" \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is nmd001, I work at NebulonMD"}'

# Semantic search
curl "http://localhost:9696/api/NebulonMind/search?q=NebulonMD&user_id=nmd001"

# Store memory directly (with lifecycle gate)
curl -X POST "http://localhost:9696/api/NebulonMind/memory?user_id=nmd001&gate=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Important fact", "category": "fact"}'

# Store memory with custom language and type
curl -X POST "http://localhost:9696/api/NebulonMind/memory?user_id=nmd001&gate=true" \
  -H "Content-Type: application/json" \
  -d '{"text": "Document in Spanish", "lang": "es", "memory_type": "doc"}'

# Run background consolidation manually
curl -X POST "http://localhost:9696/api/NebulonMind/background/memory/run?user_id=nmd001" \
  -H "Content-Type: application/json" \
  -d '{"mode": "consolidate"}'

# Run weekly summary manually
curl -X POST "http://localhost:9696/api/NebulonMind/background/task/run?user_id=nmd001" \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'

# Check LLM status
curl "http://localhost:9696/api/NebulonMind/llm/status"
```

Interactive OpenAPI docs: `http://localhost:9696/docs`

### NebulonDB Dashboard

Open **<http://localhost:6969/api/NebulonDB/dashboard/>** to view raw documents in NebulonDB.

The segment data table now displays `lang` and `type` fields for each record:
- `lang` — language code (e.g., `en`, `es`) from the `lang_type` parameter
- `type` — document type (`chat`, `doc`, `other`) mapped from `memory_type`

This helps verify that documents are stored with the correct metadata.

### How `/agent/chat` Works (Internal Pipeline)

The chat endpoint **does not call other HTTP endpoints** — it runs the full pipeline in-process:

```
User text → /agent/chat
    │
    ├─► LLM Provider (from nmd_llm_provider + nmd_llm_model)
    │
    ├─► Tools available to LLM:
    │       • recall(query)  ──► /search logic (vector + graph + ranking)
    │       • remember(text) ──► /intelligence/process logic (extract → gate → store)
    │       • decide(text)   ──► /intelligence/decide logic (extract only)
    │
    └─► Returns final answer + trace
```

**So `/search`, `/intelligence/decide`, `/intelligence/process` are already used inside chat** — you don't call them separately.

### Background Agents — Automatic + Manual

The scheduler starts automatically with the server (`nmd_host/api/server.py:328`). Jobs run on cron:

| Agent | Cron (default) | Trigger | Output |
|-------|----------------|---------|--------|
| **MemoryAgent** (consolidation) | `0 2 * * *` (daily 02:00) | Auto + Manual | Reviews/merges duplicate memories |
| **TaskAgent** (weekly summary) | `0 9 * * 0` (Sun 09:00) | Auto + Manual | Stores weekly summary as memory |
| **Auto-delete expired** | `0 3 * * *` (daily 03:00) | Auto only | Deletes TTL-expired memories |

**To run manually** (e.g., after adding lots of memories):
```bash
# Consolidate now
curl -X POST ".../background/memory/run?user_id=sathya" -d '{"mode": "consolidate"}'

# Weekly summary now
curl -X POST ".../background/task/run?user_id=sathya" -d '{"days": 7}'
```

**Results become searchable memories** — ask in chat: *"What did I work on last week?"* → recalls the TaskAgent summary memory.

### Agent Model (Separate LLM for Tool-Calling)

`/agent/chat` can use a **different LLM** than the extractor/main pipeline:

| Setting | Config Key | Env Var | Default |
|---------|------------|---------|---------|
| Agent model | `nmd_agent_model` | `NMD_AGENT_MODEL` | Falls back to `nmd_llm_model` |

**Why use a separate model?**
- **Cost**: Smaller/cheaper model (e.g., Nemotron 4B) for agent loops
- **Latency**: Faster tool-calling turns
- **Quality**: Some models excel at structured function calling

**Example:**
```ini
[llm]
nmd_llm_model = nvidia/nemotron-3.5-lightning-30b-a3b   # main (extraction, chat)

[agent]
nmd_agent_model = nvidia/nemotron-4b                     # agent tool-calling only
```

**Priority chain:**
```
nmd_agent_model → NMD_AGENT_MODEL → nmd_llm_model → NMD_LLM_MODEL → provider default
```

If both empty → uses provider hardcoded default (Nemotron 3.5 for NVIDIA).

---

## 🧠 Memory types at a glance

| Type | Meaning | Example |
|---|---|---|
| `WORKING` | current conversation | "continue that architecture idea" |
| `SHORT_TERM` | temporarily useful | "user is debugging HNSW today" |
| `LONG_TERM` | stable user info | name, skills, preferences |
| `EPISODIC` | events | "completed Phase 1" |
| `SEMANTIC` | generalized facts | "user builds AI systems" |
| `KNOWLEDGE` | external info | docs, papers |
| `DOC` | document uploads | PDFs, manuals, files |

Retention is independent of type: `permanent`, `temporary` (TTL), or `session`.

---

## 🗺️ Roadmap

- **Voice chat** — speak-to-chat with spoken answers (mic → STT → `/agent/chat`
  → TTS); chatbot (push-to-talk) and call-center (streaming + barge-in) modes.
  Voice is front-end only — no changes to the memory core.
- **CLI/TUI hardening** — promote the terminal interface out of production.
