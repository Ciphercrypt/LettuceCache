# 🥬 LettuceCache

**Context-aware semantic cache for LLMs.** Sits in front of OpenAI (or any LLM) and returns cached responses in under 30 ms — without false hits, even when identical queries mean different things in different conversations.

[![Tests](https://img.shields.io/badge/tests-52%2F52-brightgreen)](#testing)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](#building)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

---

## The Problem

Traditional semantic caching matches on query text alone:

> *"What is the cancellation policy?"* — from a hotel booking conversation  
> *"What is the cancellation policy?"* — from a gym membership conversation  

Both are identical in text and nearly identical in embedding space. A naive cache serves the wrong answer. **LettuceCache solves this by baking conversation context into every cache key.**

---

## Request Lifecycle

```
POST /query
  ↓
ContextBuilder  →  SHA-256(intent : domain : anon_user_scope : sorted_context)
  ↓
[L1] Redis exact-match lookup                    <1 ms    → HIT: return
  ↓ miss
EmbeddingClient  →  Python sidecar (all-MiniLM-L6-v2, 384-dim)
  ↓
[L2] FAISS IVF+PQ ANN search (top-5)            5–10 ms
  ↓
ValidationService  →  0.60·cosine + 0.25·ctx + 0.15·domain ≥ 0.85
  (with TurboQuant: unbiased inner-product estimation — ENABLE_TURBO_QUANT=1)
  ↓ miss
  → HIT: Templatizer::render() fills {{SLOT_N}}, backfill L1, return  <30 ms
LLM call (OpenAI gpt-4o-mini)                    500–2000 ms
  ↓ async (never blocks response)
CacheBuilderWorker
  ├─ ResponseQualityFilter  (skip conversational / session-bound / dynamic)
  └─ IntelligentAdmissionPolicy  (CVF = 0.30·freq + 0.25·cost + 0.25·quality + 0.20·novelty)
       ↓ admitted
  Templatizer  →  FAISS add  +  Redis SET  +  Redis slots SET
```

---

## Key Features

| Feature | Description |
|---|---|
| **Context isolation** | Same query from different conversations never cross-hits |
| **Two-level cache** | L1 Redis exact match (<1 ms) + L2 FAISS semantic search (5–10 ms) |
| **TurboQuant compression** | 7.8× embedding compression with provably unbiased inner products (arXiv:2504.19874) |
| **Intelligent admission** | Multi-signal Cache Value Function: frequency × cost × quality × MMR novelty |
| **Response quality filter** | Rejects conversational, session-bound, refusal, and time-sensitive responses |
| **Templatizer** | Strips UUIDs/dates/numbers into `{{SLOT_N}}` for reuse; fills slots on serve |
| **Metadata persistence** | FAISS + metadata JSON sidecar survive restarts (critical bug fix) |
| **Circuit breaker** | EmbeddingClient fails fast when Python sidecar is down |
| **Context canonicalization** | Sorted context turns → order-independent cache keys |
| **Adaptive threshold** | Per-domain admission threshold adjusts based on observed hit rates |

---

## Quick Start

```bash
# Docker Compose (Redis + Python sidecar + C++ orchestrator)
export OPENAI_API_KEY=sk-...
docker compose up

# Verify
curl http://localhost:8080/health

# First query (LLM called)
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is machine learning?","domain":"tech"}'
# → {"cache_hit": false, "latency_ms": 712, ...}

# Second query (L1 cache hit)
curl -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is machine learning?","domain":"tech"}'
# → {"cache_hit": true, "confidence": 1.0, "latency_ms": 0, ...}
```

---

## Building from Source

```bash
# macOS
brew install cmake faiss hiredis openssl curl pkg-config

# Ubuntu 22.04+
sudo apt-get install -y cmake build-essential pkg-config \
  libfaiss-dev libhiredis-dev libcurl4-openssl-dev libssl-dev

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/lettucecache
```

---

## Testing

```bash
# Unit tests (52/52, no external deps)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target unit_tests
cd build && ctest --output-on-failure

# Happy path integration test (starts stack automatically)
bash tests/happy_path.sh

# Live demo with real LLM
export OPENAI_API_KEY=sk-...
bash tests/live_demo.sh
```

---

## Configuration

All runtime config via environment variables:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | |
| `REDIS_PORT` | `6379` | |
| `EMBED_URL` | `http://localhost:8001` | Python sidecar base URL |
| `EMBED_DIM` | `384` | Must match embedding model |
| `OPENAI_API_KEY` | *(empty)* | Empty → stub mode |
| `LLM_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `HTTP_PORT` | `8080` | |
| `FAISS_INDEX_PATH` | `./faiss.index` | Also writes `.meta.json` sidecar |
| `ENABLE_TURBO_QUANT` | *(unset)* | `1` to enable TurboQuant compression |
| `CACHE_QUALITY_THRESHOLD` | `0.40` | Minimum quality score for admission |

### Tuning (requires source change)

| Parameter | Location | Default |
|---|---|---|
| Validation threshold | `src/api/HttpServer.cpp` | `0.85` |
| Scoring weights (cosine/ctx/domain) | `src/validation/ValidationService.h` | `0.60/0.25/0.15` |
| Admission CVF weights | `src/builder/IntelligentAdmissionPolicy.h` | `0.30/0.25/0.25/0.20` |
| FAISS NLIST / NPROBE / M_PQ | `src/cache/FaissVectorStore.h` | `100/10/8` |
| Admission half-life (seconds) | `IntelligentAdmissionConfig` | `120` |

---

## Intelligent Admission: Cache Value Function

The `IntelligentAdmissionPolicy` decides what gets cached using a multi-signal CVF:

```
CVF = 0.30 × frequency_score      (exponential-decay recency weighting)
    + 0.25 × generation_cost       (token_count × model_tier multiplier)
    + 0.25 × quality_score         (from ResponseQualityFilter)
    + 0.20 × novelty_score         (MMR: 1 − max_cosine_to_existing_cache)
```

**Key behaviours:**
- An expensive GPT-4 response (500 tokens) can be admitted after just **1 request**
- Near-duplicate responses (cosine > 0.94) are **hard-rejected** to prevent cache pollution
- Per-domain adaptive threshold: domains with >30% hit rate get a relaxed threshold

---

## TurboQuant (ENABLE_TURBO_QUANT=1)

Implements arXiv:2504.19874. Two-stage encoding:

1. **MSE stage (3 bits):** Randomized Hadamard Transform → Lloyd-Max scalar quantization  
2. **QJL stage (1 bit):** sign(S·residual) corrects inner-product bias

Result: **244 bytes/vector** vs 1536 bytes FP32 (6.3×), with **zero bias** on cosine estimates. The validation threshold of 0.85 remains statistically valid at 4-bit compression (σ ≈ 0.027).

---

## Architecture

```
src/
├── api/            HttpServer (composition root, all wiring)
├── orchestrator/   QueryOrchestrator, ContextBuilder, ContextSignature
├── cache/          FaissVectorStore (IVectorStore), RedisCacheAdapter
├── embedding/      EmbeddingClient (persistent CURL, circuit breaker)
├── validation/     ValidationService (TurboQuant-aware cosine)
├── builder/        CacheBuilderWorker, IntelligentAdmissionPolicy,
│                   ResponseQualityFilter, AdmissionController, Templatizer
├── quantization/   TurboQuantizer (MSE + QJL)
└── llm/            LLMAdapter, OpenAIAdapter
```

**Composition root:** `HttpServer` owns all component lifetimes as `unique_ptr`. Everything else receives references.

**Async write path:** `CacheBuilderWorker` runs on a background thread with a `std::condition_variable` queue. The orchestrator calls `enqueue()` and returns the HTTP response without waiting.

**Thread safety:** `FaissVectorStore` uses `std::shared_mutex` (concurrent reads / exclusive writes). `RedisCacheAdapter` uses a single connection (mutex needed — known open issue).

---

## API

| Endpoint | Description |
|---|---|
| `POST /query` | Main cache lookup + LLM fallback |
| `GET /health` | Dependency health (Redis, sidecar, FAISS count) |
| `GET /stats` | FAISS entry count + queue depth |
| `DELETE /cache/:key` | Evict a specific entry |

See [full API docs](https://ciphercrypt.github.io/LettuceCache/) for request/response schemas.

---

## Research References

| Paper | Relevance |
|---|---|
| arXiv:2504.19874 — TurboQuant | Vector quantization, unbiased inner products |
| MeanCache (IPDPS 2025) | Context-aware caching, federated learning |
| vCache (arXiv:2502.03771) | Per-prompt adaptive thresholds |
| SCALM (IWQoS 2024) | Semantic pattern ranking, selective admission |
| LeCaR / CACHEUS | RL-based cache replacement, decay-weighted frequency |
| CacheSack (ATC 2022) | Cost-aware admission (Google Colossus) |
| MMR (SIGIR 1998) | Diversity/novelty scoring for cache eviction |
| LHD (NSDI 2018) | Hit-density multi-signal cache value function |
