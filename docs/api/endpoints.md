# API Reference

The orchestrator exposes a REST API on port `8080` (configurable via `HTTP_PORT`).

---

## POST /query

The main entry point. Looks up the cache and falls back to the LLM on a miss.

### Request

```http
POST /query
Content-Type: application/json
```

```json
{
  "query": "What is the return policy?",
  "context": [
    "I bought a jacket last week",
    "It doesn't fit"
  ],
  "user_id": "u_123",
  "session_id": "sess_abc",
  "domain": "ecommerce",
  "correlation_id": "req_xyz"
}
```

| Field | Type | Required | Description |
|---|---|:---:|---|
| `query` | string | ✅ | The user's question |
| `context` | string[] | | Prior conversation turns. Order matters — most recent last. |
| `user_id` | string | | Anonymised to a 16-char scope token before use. Never stored raw. |
| `session_id` | string | | Passed through to logs for distributed tracing. Not used in cache logic. |
| `domain` | string | | Domain tag (e.g. `ecommerce`, `healthcare`, `general`). Used in scoring. Defaults to `general`. |
| `correlation_id` | string | | Echoed in structured logs. Useful for tracing across services. |

### Response

```json
{
  "answer": "You can return items within 30 days of purchase with original receipt.",
  "cache_hit": true,
  "confidence": 0.93,
  "cache_entry_id": "lc:l1:a3f2c1...",
  "latency_ms": 18
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | The response text |
| `cache_hit` | boolean | `true` if served from L1 or L2 cache |
| `confidence` | float | Validation score (0.0–1.0). `0.0` on LLM fallback or L1 hit (no scoring needed). |
| `cache_entry_id` | string | ID of the cache entry that was hit. Empty on LLM fallback. |
| `latency_ms` | integer | Total server-side processing time in milliseconds |

### Error Responses

| Status | Cause |
|---|---|
| `400` | Missing `query` field, or malformed JSON |
| `500` | Internal error (logged with `correlation_id`) |

---

## GET /health

Dependency health check. Used as the Kubernetes readiness probe.

```http
GET /health
```

### Response — Healthy

```http
HTTP/1.1 200 OK
```

```json
{
  "status": "ok",
  "redis": true,
  "embedding_sidecar": true,
  "faiss_entries": 1042,
  "queue_depth": 3
}
```

### Response — Degraded

```http
HTTP/1.1 503 Service Unavailable
```

```json
{
  "status": "degraded",
  "redis": false,
  "embedding_sidecar": true,
  "faiss_entries": 1042,
  "queue_depth": 3
}
```

| Field | Description |
|---|---|
| `status` | `ok` when all dependencies healthy; `degraded` otherwise |
| `redis` | Whether Redis is reachable (PING check) |
| `embedding_sidecar` | Whether the Python sidecar responds to `/health` |
| `faiss_entries` | Number of vectors currently in the FAISS index |
| `queue_depth` | Entries waiting in the `CacheBuilderWorker` queue |

---

## GET /stats

Lightweight stats snapshot. No dependency checks — always fast.

```http
GET /stats
```

```json
{
  "faiss_entries": 1042,
  "queue_depth": 3
}
```

---

## DELETE /cache/:key

Evicts a specific entry from both FAISS and the Redis L1 store.

```http
DELETE /cache/{key}
```

```bash
curl -X DELETE http://localhost:8080/cache/a3f2c1d8e9f0b2c3
```

### Response — Deleted

```http
HTTP/1.1 200 OK
```

```json
{
  "deleted": true,
  "key": "a3f2c1d8e9f0b2c3"
}
```

### Response — Not Found

```http
HTTP/1.1 404 Not Found
```

```json
{
  "deleted": false,
  "key": "a3f2c1d8e9f0b2c3"
}
```

Use this endpoint to:

- Remove a bad or outdated cache entry reported via user feedback
- Force a cache miss for a specific query pattern during testing
- Evict entries that contain stale information after a data change

---

## Python Sidecar Endpoints

The embedding sidecar runs on port `8001` (internal; not exposed to clients directly).

### POST /embed

```json
{ "text": "What is the return policy?" }
```

```json
{ "embedding": [0.023, -0.147, 0.891, ...] }
```

### POST /embed_batch

```json
{ "texts": ["query one", "query two", "query three"] }
```

```json
{ "embeddings": [[...], [...], [...]] }
```

### GET /health (sidecar)

```json
{ "status": "ok", "model": "all-MiniLM-L6-v2", "dim": 384 }
```
