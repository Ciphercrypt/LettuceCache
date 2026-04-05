# LettuceCache

**Context-aware semantic cache for LLMs.** Sits in front of OpenAI (or any LLM) and returns cached responses in 1–60 ms — without false hits, even when identical queries mean different things in different conversations.

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)](#building)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

---

## The Problem

Traditional semantic caching matches on query text alone:

> *"What is the cancellation policy?"* — from a hotel booking conversation
> *"What is the cancellation policy?"* — from a gym membership conversation

Both are identical in text and nearly identical in embedding space. A naive cache serves the wrong answer. LettuceCache solves this by baking the full deployment context — system prompt, model, temperature, tools, response format, conversation history — into every cache key.

---

## Request Lifecycle

```
POST /query
  │
  ├─ [Guard] High-temperature bypass: temperature ≥ 0.7 → LLM directly, no cache read/write
  │
  ├─ ContextBuilder
  │    ├─ signature_hash      = SHA-256(intent + all dims)   → L1 exact-match key
  │    └─ context_fingerprint = SHA-256(all dims – intent)   → L2 context validation
  │
  ├─ [L1] Redis GET lc:l1:{signature_hash}          ~1–3 ms
  │         HIT → return immediately
  │
  ├─ EmbeddingClient → Python sidecar (all-MiniLM-L6-v2, 384-dim)  ~20–50 ms
  │         dimension validated against EMBED_DIM on every call
  │
  ├─ [L2] FAISS search (IndexFlatIP below 3 900 vectors; IVF+PQ above)  ~1–3 ms
  │    └─ for each candidate:
  │         ├─ tombstone check → skip deleted entries immediately
  │         └─ ValidationService score:
  │              0.60 × cosine_sim(query_embedding, candidate_embedding)
  │            + 0.25 × (context_fingerprint == candidate.context_signature ? 1 : 0)
  │            + 0.15 × (domain match ? 1 : 0)
  │            ≥ threshold(domain)   [default 0.85; per-domain via DOMAIN_THRESHOLDS]
  │              HIT → Templatizer::render() fills {{SLOT_N}} → backfill L1 → return
  │
  └─ Cache miss → LLM call → return answer
       │
       └─ [Async] CacheBuilderWorker (background thread, never blocks response)
            ├─ AdmissionController: min 2 requests in 300s window
            ├─ ResponseQualityFilter: reject conversational / session-bound /
            │    refusal / time-sensitive responses (domain-aware whitelist)
            └─ IntelligentAdmissionPolicy CVF:
                 0.30 × frequency_score  (exponential-decay recency)
               + 0.25 × cost_score       (token_count × model_tier)
               + 0.25 × quality_score    (from ResponseQualityFilter)
               + 0.20 × novelty_score    (MMR: 1 − max_cosine_to_existing_cache)
               ≥ adaptive_threshold(domain)
                 ADMIT → Templatizer → FAISS add → Redis MULTI/EXEC (L1 + slots)
                        → domain index SADD
```

---

## Cache Key Design

All LLM call parameters that affect response uniqueness are in the key. Parameters are split into two categories:

**Framing parameters** — change what kind of response is possible. Included exactly (or as a short SHA-256 prefix):

| Parameter | Key representation |
|---|---|
| `system_prompt` | `sha256(system_prompt)[:16]` or `"none"` |
| `response_format` | `"text"` / `"json_object"` / `"json_schema:{schema_hash}"` |
| `tools` | `sha256(tools_json)[:16]` or `"none"` |
| `tool_choice` | exact string, default `"auto"` |

**Distribution parameters** — shift sampling distribution. Bucketed so adjacent values share cache partitions:

| Parameter | Bucketing |
|---|---|
| `temperature` | rounded to 1 decimal place |
| `top_p` | rounded to 1 decimal place (default 1.0) |
| `max_tokens` | `"none"` (0/unset) / `"short"` (≤200) / `"medium"` (≤800) / `"long"` (>800) |
| `seed` | exact integer, or `"none"` when absent |
| `model` | exact string |

**Two hashes serve two roles:**

- `signature_hash` = SHA-256(intent + all dims above + ordered context turns) — **L1 exact-match key**. Two requests must agree on everything, including how the question is phrased, to share an L1 entry.

- `context_fingerprint` = SHA-256(all dims above + ordered context turns, **excluding intent**) — **L2 context validation**. Two differently-phrased questions in the same deployment context (same system prompt, domain, user, model, format, tools, conversation history) share this fingerprint. FAISS cosine similarity handles whether the query content is semantically close enough.

Without this split, the maximum L2 validation score for a paraphrased query (different intent words, same meaning) is `0.60 + 0.15 = 0.75`, which is below the 0.85 threshold — semantic search would never help. With the split, a paraphrased query with 0.92 cosine similarity scores `0.60×0.92 + 0.25×1.0 + 0.15×1.0 = 0.952`, a clean hit.

---

## Key Features

| Feature | Detail |
|---|---|
| **Full LLM parameter keying** | system_prompt, response_format, tools, top_p, max_tokens, seed — all in the key |
| **L1/L2 hash split** | `signature_hash` for exact L1; `context_fingerprint` for semantic L2 — paraphrased queries hit L2 |
| **Two-level cache** | L1 Redis exact match (1–3 ms) + L2 FAISS semantic search (25–60 ms) |
| **Flat→IVF+PQ migration** | IndexFlatIP (exact, 100% recall) below 3 900 vectors; auto-migrates to IVF+PQ above |
| **TurboQuant compression** | 6.3× embedding compression with unbiased inner products (arXiv:2504.19874) |
| **Intelligent admission** | CVF: frequency × cost × quality × MMR novelty; adaptive per-domain threshold |
| **Domain-aware quality filter** | Rejects conversational, session-bound, refusal, time-sensitive responses; domain whitelist for regulated domains |
| **Atomic slot writes** | Redis MULTI/EXEC writes L1 + slot key together; slot TTL 2× L1 prevents `{{SLOT_N}}` leaks |
| **Tombstone-first DELETE** | Tombstone written before FAISS/Redis removal; concurrent L2 reads skip tombstoned entries |
| **Correct DELETE cleanup** | `find()` retrieves sig_hash + domain before removal; L1 key + slot key deleted correctly |
| **Domain bulk invalidation** | `DELETE /cache/domain/:domain` removes all entries + L1 keys + slot keys for a domain |
| **Thread-safe Redis** | `std::mutex` on single `redisContext*`; all public methods lock before `redisCommand` |
| **Embedding dim validation** | Validates sidecar response `"dimension"` field; catches silent model swaps before FAISS corruption |
| **Per-domain thresholds** | `DOMAIN_THRESHOLDS` env var (`{"faq":0.75,"compliance":0.92}`) |
| **High-temperature bypass** | `temperature ≥ 0.7` skips cache read and write entirely |
| **Circuit breaker** | EmbeddingClient fails fast (5 failures → OPEN, 30s reset) when sidecar is down |
| **FAISS persistence** | Binary index + JSON sidecar (`*.meta.json`) survive restarts |

---

## Quick Start

### Local dev (recommended)

Prerequisites: `cmake`, `faiss`, `hiredis`, `openssl`, `curl` via brew, and Python 3.9+ with a venv in `python_sidecar/.venv`.

```bash
# Build once
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Set up the Python sidecar venv once
cd python_sidecar && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cd ..

# Start everything (Redis + sidecar + binary)
export OPENAI_API_KEY=sk-...   # optional — omit for stub mode
./dev.sh

# Verify
curl http://localhost:8080/health

# Stop everything
./dev.sh stop
```

`dev.sh` manages all three processes. Logs go to `.dev-logs/`. Kills by port on stop so it works even if the binary was restarted mid-session.

### Run the smoke test

```bash
./demo.sh
```

Covers 9 scenarios end-to-end:

| Scenario | What it verifies |
|---|---|
| Admission gate | Cache is cold for 2 requests, then writes |
| L1 exact hit | 0 ms Redis hit on identical query |
| L2 semantic hit | Paraphrased queries hit at 0.89–0.95 confidence |
| Domain isolation | Same query, different domain = miss |
| System-prompt isolation | Different persona = different cache namespace |
| High-temp bypass | `temperature >= 0.7` never touches cache |
| DELETE single entry | Tombstone + FAISS eviction |
| Domain bulk invalidation | `DELETE /cache/domain/:name` clears all entries |

### Docker Compose

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
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
# Unit tests (no external deps)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target unit_tests
cd build && ctest --output-on-failure

# Integration tests (requires live Redis on localhost:6379)
docker compose up -d redis
INTEGRATION_TESTS=1 cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target integration_tests
cd build && ctest -R integration --output-on-failure
```

---

## Configuration

All runtime config via environment variables:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `EMBED_URL` | `http://localhost:8001` | Python sidecar base URL |
| `EMBED_DIM` | `384` | Must match embedding model output dimension |
| `OPENAI_API_KEY` | *(empty)* | Empty → LLM stub mode |
| `LLM_MODEL` | `gpt-4o-mini` | Default model when request omits `model` field |
| `HTTP_PORT` | `8080` | Server listen port |
| `FAISS_INDEX_PATH` | `./faiss.index` | FAISS binary; sidecar at `{path}.meta.json` |
| `ENABLE_TURBO_QUANT` | *(unset)* | Set to `1` to enable TurboQuant compression |
| `TURBO_ROTATION_SEED` | `42` | Randomized Hadamard Transform seed |
| `TURBO_QJL_SEED` | `137` | QJL Gaussian matrix seed |
| `CACHE_QUALITY_THRESHOLD` | `0.40` | Minimum quality score for admission |
| `DOMAIN_THRESHOLDS` | *(unset)* | JSON per-domain validation threshold overrides |

**Per-domain threshold example:**
```bash
export DOMAIN_THRESHOLDS='{"faq":0.75,"compliance":0.92,"banking":0.88}'
```

### Tuning (source change required)

| Parameter | File | Default |
|---|---|---|
| Global validation threshold | `src/api/HttpServer.cpp` | `0.85` |
| Scoring weights (cosine/ctx/domain) | `src/validation/ValidationService.h` | `0.60/0.25/0.15` |
| Admission CVF weights | `src/builder/IntelligentAdmissionPolicy.h` | `0.30/0.25/0.25/0.20` |
| High-temperature bypass threshold | `src/orchestrator/QueryOrchestrator.h` | `0.7` |
| FAISS IVF migration threshold | `src/cache/FaissVectorStore.h` (`MIN_IVF_TRAIN_VEC`) | `3900` |
| FAISS NLIST / NPROBE / M_PQ | `src/cache/FaissVectorStore.h` | `100/10/8` |
| Admission frequency window | `src/api/HttpServer.cpp` (`AdmissionController`) | `2 hits / 300s` |
| Admission CVF base threshold | `IntelligentAdmissionConfig` | `0.42` |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Main cache lookup + LLM fallback |
| `/health` | GET | Redis + sidecar health, FAISS entry count, queue depth |
| `/stats` | GET | FAISS entry count + builder queue depth |
| `/cache/:entry_id` | DELETE | Tombstone-first eviction; cleans up L1 + slot keys |
| `/cache/domain/:domain` | DELETE | Bulk-evict all entries for a domain |

### POST /query — Request Schema

```json
{
  "query":           "What are the overdraft fees?",
  "domain":          "banking",
  "user_id":         "user-123",
  "context":         ["I have a checking account"],
  "correlation_id":  "req-abc",
  "session_id":      "sess-xyz",

  "system_prompt":   "You are a banking assistant. Never discuss competitors.",
  "response_format": "text",
  "response_schema": {},
  "tools":           [],
  "tool_choice":     "auto",

  "model":           "gpt-4o-mini",
  "temperature":     0.2,
  "top_p":           1.0,
  "max_tokens":      500,
  "seed":            null
}
```

### POST /query — Response Schema

```json
{
  "answer":         "Our standard overdraft fee is $34 per transaction...",
  "cache_hit":      true,
  "confidence":     0.952,
  "cache_entry_id": "a3f9b2c1d4e5f678",
  "latency_ms":     47
}
```

---

## Architecture

```
src/
├── api/            HttpServer         — composition root; owns all component lifetimes
├── orchestrator/   QueryOrchestrator  — request pipeline
│                   ContextBuilder     — signature_hash + context_fingerprint
│                   ContextSignature   — SHA-256 helpers
├── cache/          FaissVectorStore   — IVF+PQ / flat two-phase vector store
│                   RedisCacheAdapter  — mutex-protected hiredis wrapper
│                   IVectorStore       — abstract interface
├── embedding/      EmbeddingClient    — persistent CURL handle + circuit breaker
├── validation/     ValidationService  — cosine + context + domain scoring
├── builder/        CacheBuilderWorker — async admission + write pipeline
│                   IntelligentAdmissionPolicy — CVF with shared_mutex
│                   ResponseQualityFilter      — domain-aware quality scoring
│                   AdmissionController        — frequency gate
│                   Templatizer                — slot extraction + render
├── quantization/   TurboQuantizer     — MSE + QJL vector compression
└── llm/            LLMAdapter / OpenAIAdapter
```

**Composition root:** `HttpServer` owns all component lifetimes as `unique_ptr`. Every other component receives references; no component creates its own dependencies.

**Async write path:** `CacheBuilderWorker` runs on a single background thread with a `std::condition_variable` queue. `QueryOrchestrator::process()` calls `enqueue()` and returns the HTTP response without waiting. All admission and quality decisions happen on the worker thread.

**Thread safety:**
- `FaissVectorStore`: `std::shared_mutex` — N concurrent `search()` readers, exclusive `add()`/`remove()`/`persist()` writers
- `RedisCacheAdapter`: `std::mutex` — single `redisContext*` protected; hiredis is not thread-safe
- `IntelligentAdmissionPolicy`: single `std::shared_mutex` — shared locks for read-only signals, exclusive for writes; eliminates dual-mutex deadlock risk
- `EmbeddingClient`: `std::mutex` on the persistent CURL handle

---

## TurboQuant (ENABLE_TURBO_QUANT=1)

Implements arXiv:2504.19874. Two-stage encoding for d=384 vectors:

1. **MSE stage (3 bits):** Randomized Hadamard Transform → Lloyd-Max scalar quantization
2. **QJL stage (1 bit):** `sign(S·residual)` corrects inner-product bias

Result: **244 bytes/vector** vs 1536 bytes FP32 (6.3×), with **zero bias** on cosine estimates (`E[TQ_ip(y, encode(x))] = ⟨y, x⟩`). The validation threshold of 0.85 remains statistically valid at 4-bit compression (σ ≈ 0.027).

Seeds are configurable via `TURBO_ROTATION_SEED` and `TURBO_QJL_SEED` for security rotation or multi-deployment isolation.

---

## Research References

| Paper | Application |
|---|---|
| arXiv:2504.19874 — TurboQuant | Vector quantization, unbiased inner products |
| MeanCache (IPDPS 2025) | Context-aware caching baseline |
| vCache (arXiv:2502.03771) | Per-prompt adaptive thresholds |
| SCALM (IWQoS 2024) | Semantic pattern ranking; configuration namespace insight |
| Semantic Caching for LLMs (Bang et al. 2023) | System prompt as first-class key dimension |
| LeCaR / CACHEUS | Decay-weighted frequency scoring |
| CacheSack (ATC 2022) | Cost-aware admission |
| MMR (SIGIR 1998) | Novelty/diversity scoring |
| LHD (NSDI 2018) | Multi-signal cache value function |
