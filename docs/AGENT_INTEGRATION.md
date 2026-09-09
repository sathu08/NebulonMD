# Agent Integration (NebulonMind as a memory layer)

NebulonMind is an HTTP service (port `9696` by default, path prefix
`/api/NebulonMind`) that any LLM-based agent can use as **its memory
backend**. This guide covers the three idiomatic integration patterns and
the honest caveats you should design around.

> **⚠️ CLI/TUI is under production** — the terminal interface (`nebulonmind.py` /
> `nmd_host/tui/`) is not yet stable for production use. Please use the **web
> console** at `http://localhost:9696/api/NebulonMind/dashboard/` instead.

## Related documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full architecture diagram, layers, domain model, persistence, agent runtime, LLM providers, background agents, configuration |
| [FILE_STRUCTURE.md](FILE_STRUCTURE.md) | Complete project file reference — top-level files, test suite, docs/, nmd_host/ package, tests/, key env vars, quick-start commands |

## Prerequisites

Every request is scoped to a *registered username* passed as the `user_id`
query parameter. **Users must be created manually** — either via the web
console or the API:

```bash
# 1. register a username (idempotent) via API
curl -X POST http://localhost:9696/api/NebulonMind/user/create_user \
  -H 'Content-Type: application/json' \
  -d '{"username": "my_agent"}'

# 2. every route expects the username as a query parameter
curl 'http://localhost:9696/api/NebulonMind/search?query=python&user_id=my_agent'
```

There is **no auto-registration**: an unknown username gets `403`.
Authentication is **optional**: set `NMD_API_AUTH_TOKEN` to require
`Authorization: Bearer <token>` on every API call (health/metrics/OpenAPI
stay public). Isolation is data partitioning, not security; protect the port yourself.

## Pattern 1 — Delegate chat (agent runtime)

Use NebulonMind's own tool-calling runtime. It grounds the answer in stored
memory, and can `remember` / `recall` context on its own. Requires an LLM
provider (`NMD_LLM_PROVIDER` + matching key/model; use `other` with
`NMD_LLM_BASE_URL` for any custom OpenAI-compatible endpoint — see
[ARCHITECTURE.md](ARCHITECTURE.md) "LLM providers").

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/agent/chat?user_id=my_agent' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What do I know about nebulon_user?",
    "session_id": "sess_abc"          # optional; ties this turn to a session
  }'
```

- Keep multi-turn coherent by passing back `transcript` from the previous
  response as `messages`, or by creating a session
  (`POST /api/NebulonMind/agent/session`) and sending its `session_id`.
- Sessions are **in-memory only** (bounded, TTL).

## Pattern 2 — Memory backend (your prompt, our recall)

Your agent does the reasoning; NebulonMind only stores and recalls. This
pattern is LLM-agnostic and needs no NebulonMind LLM config.

**Write** — always with `?gate=true` so re-seeding is idempotent instead of
creating duplicates:

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/memory?gate=true&user_id=my_agent' \
  -H 'Content-Type: application/json' \
  -d '{"content": {"text": "nebulon_user works with Python and NebulonDB."}}'
```

- `201` → newly stored.
- `200` → duplicate of an existing memory; the response body's
  `data.memory.memory_id` is the **canonical** (existing) memory id. Treat
  the call as idempotent.
- `422` → the gate refused it (e.g. an already-expired memory).

**Read** — bounded, provenance-carrying context for injecting into your
system prompt:

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/memory/context?query=nebulon_user%20python&top_k=5&max_characters=4000&user_id=my_agent' \
  -H 'Content-Type: application/json' -d '{}'
```

Or raw results for your own formatting:

```bash
curl 'http://localhost:9696/api/NebulonMind/search?query=nebulon_user%20python&top_k=5&user_id=my_agent'
```

Then interpolate `data.context` (or `data.results`) into your prompt as
"relevant memories: …".

## Pattern 3 — Hybrid (delegate turn-taking, extract durable facts)

Best of both: use `agent/chat` for conversational turn-taking, and extract
durable facts from the transcript yourself (or via the intelligence
pipeline) so they survive long after a session ends.

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/intelligence/process?user_id=my_agent' \
  -H 'Content-Type: application/json' \
  -d '{"text": "My name is nebulon_user. I work with Python.", "persist": true}'
```

`data.decisions` are the facts the engine found; `data.ingestions` report
what the Step 3 gate did with each one (`STORE` / `DUPLICATE` / `EXPIRED` /
`INVALID`). `STORE`-approved memories land in NebulonDB and are durable.

For full determinism (no provider dependency), skip the extraction pipeline
and classify the fact yourself, then write it with
`POST /memory?gate=true`.

## Reference

| Intent | Endpoint |
|---|---|
| Register username | `POST /api/NebulonMind/user/create_user` |
| Store (gated) | `POST /api/NebulonMind/memory?gate=true` |
| Store (raw) | `POST /api/NebulonMind/memory` |
| Recall | `GET /api/NebulonMind/search` |
| Bounded LLM context | `POST /api/NebulonMind/memory/context` |
| Extract + persist facts | `POST /api/NebulonMind/intelligence/process` |
| Delegate chat | `POST /api/NebulonMind/agent/chat` |
| Agent sessions | `POST/GET/DELETE /api/NebulonMind/agent/session{s, /{id}}` |
| Health | `GET /api/NebulonMind/health` |