# Agent Integration (NebulonMind as a memory layer)

NebulonMind is an HTTP service (port `9696` by default, path prefix
`/api/NebulonMind`) that any LLM-based agent can use as **its memory
backend**. This guide covers the three idiomatic integration patterns and
the honest caveats you should design around.

## Prerequisites

Every request is scoped to a *registered username* passed as the `user_id`
query parameter. Register once, then reuse the same name everywhere:

```bash
# 1. register a username (idempotent)
curl -X POST http://localhost:9696/api/NebulonMind/user/create_user \
  -H 'Content-Type: application/json' \
  -d '{"username": "my_agent"}'

# 2. every route expects the username as a query parameter
curl 'http://localhost:9696/api/NebulonMind/search?query=python&user_id=my_agent'
```

There is **no auto-registration**: an unknown username gets `403`.
Authentication is **optional**: set `NMD_API_AUTH_TOKEN` to require
`Authorization: Bearer <token>` on every API call (health/metrics/OpenAPI
stay public). See [Caveats](#caveats) — isolation is data partitioning, not
security; protect the port yourself.

## Pattern 1 — Delegate chat (agent runtime)

Use NebulonMind's own tool-calling runtime. It grounds the answer in stored
memory, and can `remember` / `recall` context on its own. Requires an LLM
provider (`NMD_LLM_PROVIDER` + matching key/model).

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/agent/chat?user_id=my_agent' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What do I know about Sathya?",
    "session_id": "sess_abc"          # optional; ties this turn to a session
  }'
```

- Keep multi-turn coherent by passing back `transcript` from the previous
  response as `messages`, or by creating a session
  (`POST /api/NebulonMind/agent/session`) and sending its `session_id`.
- Sessions are **in-memory only** (bounded, TTL). See caveat 2.

## Pattern 2 — Memory backend (your prompt, our recall)

Your agent does the reasoning; NebulonMind only stores and recalls. This
pattern is LLM-agnostic and needs no NebulonMind LLM config.

**Write** — always with `?gate=true` so re-seeding is idempotent instead of
creating duplicates:

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/memory?gate=true&user_id=my_agent' \
  -H 'Content-Type: application/json' \
  -d '{"content": {"text": "Sathya works with Python and NebulonDB."}}'
```

- `201` → newly stored.
- `200` → duplicate of an existing memory; the response body's
  `data.memory.memory_id` is the **canonical** (existing) memory id. Treat
  the call as idempotent.
- `422` → the gate refused it (e.g. an already-expired memory).

**Read** — bounded, provenance-carrying context for injecting into your
system prompt:

```bash
curl -X POST 'http://localhost:9696/api/NebulonMind/memory/context?query=Sathya%20python&top_k=5&max_characters=4000&user_id=my_agent' \
  -H 'Content-Type: application/json' -d '{}'
```

Or raw results for your own formatting:

```bash
curl 'http://localhost:9696/api/NebulonMind/search?query=Sathya%20python&top_k=5&user_id=my_agent'
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
  -d '{"text": "My name is Sathya. I work with Python.", "persist": true}'
```

`data.decisions` are the facts the engine found; `data.ingestions` report
what the Step 3 gate did with each one (`STORE` / `DUPLICATE` / `EXPIRED` /
`INVALID`). `STORE`-approved memories land in NebulonDB and are durable.

For full determinism (no provider dependency), skip the extraction pipeline
and classify the fact yourself, then write it with
`POST /memory?gate=true`.

## Caveats

These are current-state limitations and how the service mitigates them when
you opt in:

1. **LLM extraction is model-dependent.** The rules engine is the default
   and is deterministic; the optional LLM extractor (`NMD_LLM_EXTRACTOR`)
   is not — it has returned empty documents in practice. The service now
   falls back to the rule extractor whenever the LLM **fails or returns an
   empty payload**, so a flaky model never silently loses a turn. For full
   determinism, classify facts yourself and use `POST /memory?gate=true`.

2. **Sessions are in-memory by default** (`nmd_host/agent/session.py`):
   bounded per user (`NMD_AGENT_MAX_SESSIONS`, default 100) with a TTL
   (`NMD_AGENT_SESSION_TTL_SECONDS`, default 3600). A NebulonMind restart
   clears the registry. **Durable facts are not lost** — stored memories
   live in NebulonDB. To survive restarts *beyond memory*, set
   `NMD_AGENT_DURABLE_SESSIONS=true`: sessions are persisted in a
   NebulonDB corpus (`mind_sessions`) and re-hydrated on the next
   `session_id` access. For critical flows you can also pass your own
   `messages` history instead of relying on server sessions.

3. **Raw `POST /memory` bypasses the de-dup gate.** Only `?gate=true`
   routes through the Step 3 ingest gate (duplicate / expiry / retention).
   External seeders should always set `gate=true` to avoid re-seeding
   duplicates.

4. **Background jobs are single-node** (`nmd_host/agents/scheduler.py`):
   an in-process scheduler whose run history is in-memory and resets on
   restart. Set `NMD_BACKGROUND_PERSIST_STATE=true` to persist
   `last_run`/`last_result`/`runs` in a NebulonDB corpus (`mind_background`)
   so job history survives restarts. If you need truly multi-node
   scheduling, trigger `POST /background/memory/run` /
   `POST /background/task/run` yourself.

5. **Auth is optional and off by default.** Isolation is per registered
   username (`user_id`), not credentials — treat it as data partitioning.
   Set `NMD_API_AUTH_TOKEN=<secret>` to require
   `Authorization: Bearer <secret>` (or `X-API-Key: <secret>`) on every
   route; health, `/metrics` and OpenAPI stay public. Put the service
   behind your own auth proxy for defense in depth. Rate limiting is per
   client IP only.

6. **TEMPORARY facts auto-expire.** TEMPORARY retention without an explicit
   `expires_at` gets a default expiry of `NMD_TEMPORARY_TTL_SECONDS`
   (default 30 days) when stored through the gate. For durable facts set
   `lifecycle.retention_policy: "permanent"` (or `"long_term"`); expiry is
   only enforced for TEMPORARY memories.

## Reference

| Intent | Endpoint |
| --- | --- |
| Register username | `POST /api/NebulonMind/user/create_user` |
| Store (gated) | `POST /api/NebulonMind/memory?gate=true` |
| Store (raw) | `POST /api/NebulonMind/memory` |
| Recall | `GET /api/NebulonMind/search` |
| Bounded LLM context | `POST /api/NebulonMind/memory/context` |
| Extract + persist facts | `POST /api/NebulonMind/intelligence/process` |
| Delegate chat | `POST /api/NebulonMind/agent/chat` |
| Agent sessions | `POST/GET/DELETE /api/NebulonMind/agent/session{s, /{id}}` |
| Health | `GET /api/NebulonMind/health` |