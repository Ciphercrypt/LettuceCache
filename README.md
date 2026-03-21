# LettuceCache

A context-aware semantic caching system that sits in front of LLMs to reduce API cost and latency. Unlike traditional caches that match on query text alone, LettuceCache encodes conversation context into the cache key — preventing false hits when the same question means different things in different conversations.

---

## How It Works

Every query goes through a two-layer lookup before the LLM is ever called:

```
POST /query
    │
    ├─ ContextBuilder  →  intent + domain + anon user scope → SHA-256 signature
    │
    ├─ L1: Redis (exact hash match)          < 1 ms
    │      hit → return immediately
    │
    ├─ EmbeddingClient → Python sidecar      ~ 10 ms
    │
    ├─ L2: FAISS IVF+PQ (ANN search)         ~ 5–10 ms
    │      ValidationService scores candidates
    │      score = 0.60·cosine + 0.25·ctx_sig + 0.15·domain ≥ 0.85 → hit
    │
    └─ LLM fallback (OpenAI)                 ~ 500–2000 ms
           └─ async enqueue → CacheBuilderWorker
```

On a cache hit the response is returned in **< 30 ms**. On a miss the LLM is called and the result is asynchronously processed and indexed without blocking the response path.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Orchestrator — C++ HTTP server (:8080)               │
│  QueryOrchestrator drives the full hot path           │
└──────────┬───────────────────────┬───────────────────┘
           │                       │
    ┌──────▼──────┐       ┌────────▼────────────────┐
    │  Redis 7    │       │  CacheBuilderWorker      │
    │  L1 store   │       │  AdmissionController     │
    │  + Streams  │       │  Templatizer             │
    └─────────────┘       │  → FAISS + Redis + PG    │
                          └─────────────────────────-┘
    ┌──────────────┐      ┌──────────────────────┐
    │  FAISS index │      │  Python sidecar       │
    │  IVF+PQ      │      │  sentence-transformers│
    │  persisted   │      │  FastAPI  (:8001)     │
    └──────────────┘      └──────────────────────┘
```

### Component Map

| Component | File | Role |
|---|---|---|
| `QueryOrchestrator` | `src/orchestrator/QueryOrchestrator.cpp` | Drives the full L1 → embed → L2 → LLM hot path |
| `ContextBuilder` | `src/orchestrator/ContextBuilder.cpp` | Extracts intent (first 3 non-stopword tokens) and builds `ContextObject` |
| `ContextSignature` | `src/orchestrator/ContextSignature.cpp` | `SHA-256(intent:domain:anon_user_scope)` — the anonymised cache key |
| `RedisCacheAdapter` | `src/cache/RedisCacheAdapter.cpp` | hiredis wrapper; `SETEX` get/set/del + Redis Streams XADD/XREAD |
| `FaissVectorStore` | `src/cache/FaissVectorStore.cpp` | IVF+PQ index; lazy-trains at 256 vectors; persist/load on disk |
| `ValidationService` | `src/validation/ValidationService.cpp` | Weighted composite score; threshold 0.85 |
| `CacheBuilderWorker` | `src/builder/CacheBuilderWorker.cpp` | Background thread with `condition_variable` queue — writes never block reads |
| `AdmissionController` | `src/builder/AdmissionController.cpp` | Sliding-window frequency gate (default: seen ≥ 2× in 5 min) |
| `Templatizer` | `src/builder/Templatizer.cpp` | Replaces high-entropy tokens (UUIDs, IDs) with `{{SLOT_N}}` |
| `EmbeddingClient` | `src/embedding/EmbeddingClient.cpp` | libcurl POST to Python sidecar; caches vectors by query hash in Redis |
| `OpenAIAdapter` | `src/llm/OpenAIAdapter.cpp` | OpenAI Chat Completions; graceful no-key fallback for local dev |
| `HttpServer` | `src/api/HttpServer.cpp` | cpp-httplib; owns and wires all components |
| Python sidecar | `python_sidecar/main.py` | FastAPI + `sentence-transformers`; `/embed`, `/embed_batch`, `/health` |

---

## Repository Structure

```
lettucecache/
├── CMakeLists.txt                  # CMake build; lettucecache_lib shared by exe + tests
├── Dockerfile                      # Multi-stage C++ build → minimal runtime image
├── docker-compose.yml              # Redis 7 + Python sidecar + C++ orchestrator
├── .env.example                    # All config env vars documented
├── src/
│   ├── main.cpp
│   ├── orchestrator/               # Hot path coordination
│   ├── cache/                      # Redis + FAISS adapters
│   ├── validation/                 # Scoring and threshold logic
│   ├── builder/                    # Async write path
│   ├── embedding/                  # HTTP client to Python sidecar
│   ├── llm/                        # Abstract LLMAdapter + OpenAI impl
│   └── api/                        # REST endpoints
├── python_sidecar/                 # FastAPI embedding microservice
├── tests/
│   ├── unit/                       # Pure in-memory tests (no I/O)
│   └── integration/                # Requires live Redis; gated by INTEGRATION_TESTS=1
└── k8s/                            # Kubernetes Deployment, Service, ConfigMap
```

---

## API Reference

### `POST /query`

Main entry point. Returns a cached or freshly generated answer.

**Request**
```json
{
  "query": "What is the return policy?",
  "context": ["I bought a jacket last week", "It doesn't fit"],
  "user_id": "u_123",
  "session_id": "sess_abc",
  "domain": "ecommerce",
  "correlation_id": "req_xyz"
}
```

**Response**
```json
{
  "answer": "You can return items within 30 days of purchase...",
  "cache_hit": true,
  "confidence": 0.93,
  "cache_entry_id": "lc:l1:a3f2...",
  "latency_ms": 18
}
```

| Field | Type | Description |
|---|---|---|
| `query` | string | **Required.** The user's question |
| `context` | string[] | Prior conversation turns (optional) |
| `user_id` | string | Anonymised to a 16-char scope token before storage |
| `session_id` | string | Passed through for logging only |
| `domain` | string | Domain tag used in scoring (default: `general`) |
| `correlation_id` | string | Echoed in logs for tracing |

---

### `GET /health`

```json
{
  "status": "ok",
  "redis": true,
  "embedding_sidecar": true,
  "faiss_entries": 1042,
  "queue_depth": 3
}
```

Returns `200 OK` when all dependencies are healthy, `503` when degraded.

---

### `GET /stats`

```json
{
  "faiss_entries": 1042,
  "queue_depth": 3
}
```

---

### `DELETE /cache/:key`

Evicts an entry from both FAISS and the Redis L1 store.

```bash
curl -X DELETE http://localhost:8080/cache/a3f2c1...
```

```json
{ "deleted": true, "key": "a3f2c1..." }
```

---

## Validation Scoring

A FAISS candidate is returned as a cache hit only when:

```
score = 0.60 × cosine_similarity
      + 0.25 × context_signature_match   (1.0 if SHA-256 matches, else 0.0)
      + 0.15 × domain_match              (1.0 if domains match, else 0.0)
      ≥ 0.85
```

The threshold is configurable via `ValidationService(threshold)`. The context signature encodes `intent:domain:anon_user_scope` — the same query with a different conversation context produces a different signature and will miss, even if cosine similarity is high.

---

## Admission Control

Entries are not written to the vector index unconditionally. `AdmissionController` gates writes behind:

- **Minimum frequency**: query signature must be seen ≥ 2 times (configurable)
- **Time window**: within a rolling 5-minute window (configurable)
- **Max response size**: responses > 32 KB are rejected (configurable)

This prevents ephemeral or one-off queries from polluting the FAISS index.

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- C++17 compiler (clang++ 14+ or GCC 12+)
- CMake 3.20+
- System libs: `libfaiss`, `libhiredis`, `libcurl`, `libssl`

### Run with Docker Compose

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

docker compose up
```

The orchestrator will be available at `http://localhost:8080`.

### Build from Source

```bash
# Install system dependencies (macOS)
brew install faiss hiredis openssl curl cmake

# Install system dependencies (Ubuntu)
apt-get install libfaiss-dev libhiredis-dev libcurl4-openssl-dev libssl-dev cmake

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Start Redis and Python sidecar first
docker compose up redis python_sidecar

# Run
OPENAI_API_KEY=sk-... ./build/lettucecache
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `EMBED_URL` | `http://localhost:8001` | Python sidecar base URL |
| `EMBED_DIM` | `384` | Embedding dimension (must match model) |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key; omit for local/test mode |
| `HTTP_PORT` | `8080` | Port the C++ server listens on |
| `FAISS_INDEX_PATH` | `./faiss.index` | Path to persist/load the FAISS index |

---

## Running Tests

```bash
# Unit tests (no external dependencies)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test_unit
cd build && ctest -R unit --output-on-failure

# Integration tests (requires Redis on localhost:6379)
INTEGRATION_TESTS=1 cmake -B build
cmake --build build --target test_integration
cd build && ctest -R integration --output-on-failure
```

---

## Kubernetes

Manifests in `k8s/` deploy the orchestrator and Python sidecar as separate workloads. Config is injected via `ConfigMap`.

```bash
kubectl apply -f k8s/
```

The sidecar has a `/health` readiness probe to avoid cold-start misses during rollout.

---

## Technology Choices

| Concern | Choice | Why |
|---|---|---|
| Language | C++17 | Maximum performance on the hot path |
| L1 cache | Redis 7 via hiredis | Sub-millisecond exact match; O(1) |
| L2 vector search | FAISS IVF+PQ | C++ native; ~5–10 ms p95 at 1M vectors |
| HTTP server | cpp-httplib | Header-only; zero extra build complexity |
| Async writes | Redis Streams | Decouples write from read; already have Redis |
| Embedding | Python FastAPI sidecar | Easy model swap; avoids ONNX C++ complexity |
| LLM | Abstract `LLMAdapter` | Pluggable; ships with OpenAI implementation |
| Build | CMake + FetchContent | Standard; no vcpkg required |
