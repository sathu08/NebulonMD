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
| `[agent]` | `nmd_agent_temperature`, `nmd_agent_max_turns`, `nmd_agent_system_prompt` | — | agent-chat behaviour |
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

> The terminal CLI/TUI is still **under production** — please use the website
> console for day-to-day usage until it ships.

### REST API

Every route scopes to a registered username via `?user_id=`:

```bash
# create a mind user
curl -X POST "http://localhost:9696/api/NebulonMind/user/create_user?user_id=nebulon_user"

# agent chat (memory-aware tool loop: remember + recall)
curl -X POST "http://localhost:9696/api/NebulonMind/agent/chat?user_id=nebulon_user" \
  -H "Content-Type: application/json" \
  -d '{"text": "Remember that my birthday is May 5"}'

# semantic search over stored memories
curl "http://localhost:9696/api/NebulonMind/search?q=birthday&user_id=nebulon_user"
```

Interactive OpenAPI docs: `http://localhost:9696/docs`

### Use it from your own agent

Any external agent can use NebulonMind as its memory/brain through the REST
API — delegate chat to `/agent/chat`, or write/read memories directly with
`POST /memory?gate=true` and `GET /search`.

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

Retention is independent of type: `permanent`, `temporary` (TTL), or `session`.

---

## 🗺️ Roadmap

- **Voice chat** — speak-to-chat with spoken answers (mic → STT → `/agent/chat`
  → TTS); chatbot (push-to-talk) and call-center (streaming + barge-in) modes.
  Voice is front-end only — no changes to the memory core.
- **CLI/TUI hardening** — promote the terminal interface out of production.
