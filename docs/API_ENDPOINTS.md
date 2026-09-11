# NebulonMind API Documentation

## Overview

NebulonMind provides a comprehensive REST API for memory management, intelligent agent operations, and system management. The API follows RESTful conventions and returns structured JSON responses with consistent envelopes.

**Base URL:** `http://localhost:9696/api/NebulonMind`

## Response Format

All API responses follow a standard envelope structure:

```json
{
  "success": true,
  "message": "description of result",
  "data": { /* actual response data */ }
}
```

## Authentication

Currently, the API uses simple username-based authentication via query parameter:
- `user_id`: Registered NebulonMind username

---

## API Endpoints

### 1. Service Health & Metrics (4 endpoints)

#### 1.1 Liveness Probe
```http
GET /health/live
```
Check if the NebulonMind process is running.

**Response:**
```json
{
  "success": true,
  "message": "service live",
  "data": {
    "service": "NebulonMind",
    "status": "live"
  }
}
```

#### 1.2 Readiness Probe
```http
GET /health/ready
```
Check if the service can do useful work (NebulonDB connection required).

**Response:**
```json
{
  "success": true,
  "message": "service ready",
  "data": {
    "service": "NebulonMind",
    "backend": "up",
    "status": "ready"
  }
}
```

#### 1.3 Service Health
```http
GET /health
```
Compatibility endpoint that merged liveness and readiness probes.

#### 1.4 Prometheus Metrics
```http
GET /metrics
```
Prometheus metrics in text format (excluded from OpenAPI schema).

---

### 2. User Management (3 endpoints)

#### 2.1 Register Username
```http
POST /user/create_user
```
Register a new username (idempotent).

**Request:**
```json
{
  "username": "sathya"
}
```

**Response:**
```json
{
  "success": true,
  "message": "user 'sathya' created",
  "data": {
    "username": "sathya",
    "user_id": "user_123",
    "created": true
  }
}
```

#### 2.2 Resolve Username
```http
GET /user/resolve?user_id=sathya
```
Resolve a registered username to user_id (never auto-registers).

#### 2.3 Switch to Username
```http
POST /user/setup
```
Switch to an already-registered username.

**Request:**
```json
{
  "username": "sathya"
}
```

---

### 3. Memory Management (4 endpoints)

#### 3.1 Store Memory
```http
POST /memory
```
Persist a memory across truth, meaning, and relationship stores.

**`memory_type` → NebulonDB `doc_type` mapping** (stored verbatim, exactly as
the dashboard shows it): `doc` → `doc`, every other `memory_type` →
`chat_memory`. `lang` defaults to `en`. Never send chat text with a document
type — `doc`/`pdf` tags mean file origin and earn a retrieval boost.

**Request:**
```json
{
  "content": {
    "text": "My name is Sathya",
    "summary": "User's name is Sathya"
  },
  "classification": {
    "memory_type": "long_term",
    "category": "identity",
    "source": "conversation",
    "lang": "en"
  },
  "importance": {
    "score": 0.99,
    "confidence": 1.0,
    "priority": "high"
  },
  "lifecycle": {
    "retention_policy": "permanent"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "memory stored",
  "data": {
    "memory": {
      "memory_id": "mem_123",
      "user_id": "user_123",
      "content": {
        "text": "My name is Sathya",
        "summary": "User's name is Sathya"
      },
      "classification": {
        "memory_type": "long_term",
        "category": "identity",
        "source": "conversation",
        "lang": "en"
      },
      "importance": {
        "score": 0.99,
        "confidence": 1.0,
        "priority": "high"
      },
      "lifecycle": {
        "retention_policy": "permanent",
        "created_at": "2026-09-06T15:31:03.604576Z"
      },
      "status": "active"
    }
  }
}
```

#### 3.2 Get Memory
```http
GET /memory/{memory_id}
```
Retrieve a specific memory by ID.

#### 3.3 Update Memory
```http
PUT /memory/{memory_id}
```
Partially update a memory (only provided fields are replaced).

#### 3.4 Delete Memory
```http
DELETE /memory/{memory_id}
```
Delete a memory (deletes in reverse write order: graph → vector → truth).

#### 3.5 Store Document Pages (PDF Upload Pattern)
```http
POST /memory?gate=true
```
External converters (PDF → text lives **outside** NebulonMind) push **one
request per page** through the regular Store Memory endpoint — no special
upload API exists. Each page becomes a searchable `doc` memory; add one
extra request holding the whole-document summary.

**Request (one page):**
```json
{
  "user_id": "nmd_user_01",
  "content": {
    "text": "<page 3 extracted text>",
    "summary": "<one-line summary of page 3>",
    "structured_data": {"filename": "manual.pdf", "page": 3, "total_pages": 40}
  },
  "classification": {"memory_type": "doc", "category": "knowledge",
                     "source": "manual.pdf", "lang": "en"},
  "lifecycle": {"retention_policy": "permanent"}
}
```

**Field guide:**

| Field | Purpose |
|---|---|
| `content.text` | The extracted page text (required, non-empty) |
| `content.summary` | One-line page summary (listings, dedupe signals) |
| `content.structured_data` | Free-form dict, stored verbatim — `filename`, `page`, `total_pages` |
| `classification.source` | Filename — surfaces as `source:` in recall context so answers cite it |
| `memory_type: doc` | Stored with `doc_type=doc` (keep `gate=true`: re-uploads return `DUPLICATE` instead of doubling) |

**Whole-document summary (one extra request):** same shape, `text` = full-document
summary, `structured_data: {"filename": "manual.pdf", "kind": "document_summary"}`.

**Q&A follow-up:** ask via `POST /agent/chat` (or `GET /search`) — recall context
carries `source: manual.pdf`, so answers cite the file. Keep pages under ~50k
chars; upload order does not matter; re-running is idempotent.

```bash
curl -X POST "http://localhost:9696/api/NebulonMind/memory?user_id=nmd_user_01&gate=true" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "nmd_user_01",
       "content": {"text": "Page 3: install with pip install nebulonmind.",
                   "summary": "Installation command",
                   "structured_data": {"filename": "manual.pdf", "page": 3, "total_pages": 40}},
       "classification": {"memory_type": "doc", "category": "knowledge",
                          "source": "manual.pdf", "lang": "en"},
       "lifecycle": {"retention_policy": "permanent"}}'
```

---

### 4. Memory Recall & Search (2 endpoints)

#### 4.1 Semantic Recall
```http
GET /search
```
Retrieve memories through the lifecycle pipeline with optional graph expansion.

**Query Parameters:**
- `query`: Search text
- `top_k`: Number of results (default: 5)
- `expand`: Include depth-1 graph relationships (default: false)
- `user_id`: Registered username

**Response:**
```json
{
  "success": true,
  "message": "3 memories recalled",
  "data": {
    "results": [
      {
        "memory_id": "mem_123",
        "content": {
          "text": "My name is Sathya"
        },
        "classification": {
          "memory_type": "long_term",
          "category": "identity",
          "lang": "en"
        },
        "importance": {
          "score": 0.99,
          "priority": "high"
        }
      }
    ]
  }
}
```

#### 4.2 Build Context
```http
POST /memory/context
```
Build bounded LLM context for a query.

**Request:**
```json
{
  "query": "Tell me about the user",
  "top_k": 5,
  "max_characters": 2000
}
```

**Response:**
```json
{
  "success": true,
  "message": "context built",
  "data": {
    "context": "User context information..."
  }
}
```

---

### 5. Intelligence & Decision Making (2 endpoints)

#### 5.1 Decide (No Storage)
```http
POST /intelligence/decide
```
Convert conversation to memory decisions without storage.

**Request:**
```json
{
  "turns": [
    {
      "role": "user",
      "content": "My name is Sathya"
    }
  ]
}
```

#### 5.2 Process (With Storage)
```http
POST /intelligence/process
```
Decide over conversation and persist through lifecycle gate.

---

### 6. Relationship Management (1 endpoint)

#### 6.1 Link Memory to Entity
```http
POST /memory/{memory_id}/relate
```
Add a memory → entity edge in the graph.

**Request:**
```json
{
  "entity": "Python",
  "relation": "HAS_SKILL"
}
```

---

### 7. Agent Chat & Sessions (5 endpoints)

#### 7.1 Agent Chat
```http
POST /agent/chat
```
Agent chat with tool-calling loop over user's memory.

**Request:**
```json
{
  "text": "What should I learn next?",
  "messages": [
    {
      "role": "user",
      "content": "I'm learning Python"
    }
  ],
  "session_id": "optional_session_id"
}
```

#### 7.2 Create Agent Session
```http
POST /agent/session
```
Create a logical agent session.

#### 7.3 Get Agent Session
```http
GET /agent/session/{session_id}
```
Retrieve an active agent session.

#### 7.4 List Agent Sessions
```http
GET /agent/sessions
```
List user's active agent sessions.

#### 7.5 Close Agent Session
```http
DELETE /agent/session/{session_id}
```
Close and drop an agent session.

---

### 8. Evaluation & Testing (2 endpoints)

#### 8.1 Get Evaluation Dataset
```http
GET /evaluation/dataset
```
Get the bundled benchmark dataset.

#### 8.2 Run Evaluation
```http
POST /evaluation/run
```
Run agent evaluation over benchmark dataset.

**Request:**
```json
{
  "dataset": "memory_test_v1",
  "seed": true,
  "max_items": 10
}
```

---

### 9. Background Jobs (3 endpoints)

#### 9.1 Run Memory Agent
```http
POST /background/memory/run
```
Manually trigger the Memory Agent.

#### 9.2 Run Task Agent
```http
POST /background/task/run
```
Manually trigger the Task Agent.

#### 9.3 Background Status
```http
GET /background/status
```
Get background scheduler status.

---

### 10. LLM Provider Status (1 endpoint)

#### 10.1 LLM Status
```http
GET /llm/status
```
Check configured LLM provider status.

---

### 11. Chat Management (3 endpoints)

#### 11.1 List Chats
```http
GET /chats
```
List user's saved chat transcripts.

#### 11.2 Get Chat
```http
GET /chats/{chat_id}
```
Fetch a single chat transcript.

#### 11.3 Save Chat
```http
POST /chats
```
Save or overwrite a chat transcript.

**Request:**
```json
{
  "user_id": "sathya",
  "chat": {
    "id": "chat_123",
    "title": "Our conversation",
    "messages": [
      {
        "role": "user",
        "content": "Hello"
      }
    ]
  }
}
```

#### 11.4 Delete Chat
```http
DELETE /chats/{chat_id}
```
Delete a chat transcript.

---

### 12. System Management (via included routers)

#### 12.1 Dashboard Routes
- `GET /dashboard` - Serve web console
- `GET /dashboard/config` - Get configuration
- `PUT /dashboard/config` - Update configuration
- `GET /dashboard/web` - Web console alias

#### 12.2 Config Routes
- `GET /config/cfg` - Get nebulonmind.cfg settings
- `PUT /config/cfg` - Update configuration
- `POST /config/restart` - Restart service

---

## Memory Object Structure

### Core Fields
```json
{
  "memory_id": "mem_123",
  "user_id": "user_123",
  "content": {
    "text": "Memory content",
    "summary": "Short summary",
    "structured_data": {}
  },
  "classification": {
    "memory_type": "long_term|short_term|working|episodic|semantic|knowledge|doc",
    "category": "identity|preference|skill|goal|project|fact|event|task|opinion|knowledge|general",
    "source": "conversation|agent|api|import",
    "lang": "en"
  },
  "importance": {
    "score": 0.99,
    "confidence": 0.95,
    "priority": "high|medium|low"
  },
  "lifecycle": {
    "retention_policy": "permanent|temporary|session",
    "created_at": "2026-09-06T15:31:03.604576Z",
    "updated_at": "2026-09-06T15:31:03.604576Z",
    "expires_at": null
  },
  "status": "active|archived",
  "entities": ["Python", "JavaScript"],
  "relationships": [
    {
      "source": "Sathya",
      "target": "Python",
      "relation": "HAS_SKILL"
    }
  ]
}
```

### Memory Types
- **working**: Short-term, session-based memories
- **short_term**: Temporary memories with expiration
- **long_term**: Permanent memories (identity, preferences, skills)
- **episodic**: Events and experiences
- **semantic**: General knowledge and facts
- **knowledge**: Documented information
- **doc**: Document content

### Retention Policies
- **permanent**: Never expire (identity, core preferences)
- **temporary**: Auto-expire after TTL (facts, events)
- **session**: Expire when session ends (working memory)

## Error Handling

All endpoints return HTTP status codes with error details:

```json
{
  "success": false,
  "message": "error description",
  "detail": "additional error information"
}
```

Common status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Web Console

The web console provides a user interface for the API and is available at:
- `http://localhost:9696/api/NebulonMind/dashboard`

## Rate Limiting

API requests are rate-limited to prevent abuse. Configure `NMD_API_RATE_LIMIT_PER_MINUTE` in `.env`.

## CORS

Configure allowed origins with `NMD_API_CORS_ORIGINS` in `.env`.

---

## Quick Examples

### Store a memory
```bash
curl -X POST "http://localhost:9696/api/NebulonMind/memory?user_id=sathya" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {"text": "I love Python programming"},
    "classification": {"memory_type": "preference", "category": "preference", "lang": "en"},
    "importance": {"score": 0.8, "priority": "high"}
  }'
```

### Search memories
```bash
curl "http://localhost:9696/api/NebulonMind/search?query=python&user_id=sathya&top_k=5"
```

### Chat with agent
```bash
curl -X POST "http://localhost:9696/api/NebulonMind/agent/chat?user_id=sathya" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What should I learn next?",
    "messages": [{"role": "user", "content": "I know Python basics"}]
  }'
```

---

*For more detailed information about architecture and integration, see the other documentation files in the `docs/` directory.*