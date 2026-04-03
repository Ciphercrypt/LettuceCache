# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# macOS dependencies
brew install cmake faiss hiredis openssl curl pkg-config

# Ubuntu 22.04+
sudo apt-get install -y cmake build-essential pkg-config libfaiss-dev libhiredis-dev libcurl4-openssl-dev libssl-dev

# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j$(nproc)
```

`nlohmann/json`, `cpp-httplib`, `spdlog`, and `googletest` are fetched automatically by FetchContent if not found locally.

## Tests

```bash
# Unit tests (no external deps)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target unit_tests
cd build && ctest -R unit --output-on-failure

# Run a single test suite
ctest -R test_validation --output-on-failure

# Integration tests (requires live Redis on localhost:6379)
docker compose up -d redis
INTEGRATION_TESTS=1 cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target integration_tests
cd build && ctest -R integration --output-on-failure
```

## Python Sidecar

```bash
cd python_sidecar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8001
```

## Run the Full Stack

```bash
docker compose up          # starts Redis + Python sidecar + C++ orchestrator
./build/lettucecache       # or run the binary directly after setting env vars
```

## Architecture

The system is a **context-aware semantic cache** that intercepts LLM queries through two cache levels before hitting OpenAI:

```
POST /query
  ├─ High-temperature bypass (temperature ≥ 0.7 → LLM directly, no cache)
  ├─ ContextBuilder → signature_hash (L1) + context_fingerprint (L2)
  ├─ L1: Redis GET lc:l1:{signature_hash}      ~1–3 ms
  ├─ L2: FAISS search + ValidationService      ~25–60 ms
  └─ LLM fallback → async CacheBuilderWorker
```

### Key architectural constraints

**Two-hash key design** (`src/orchestrator/ContextBuilder.cpp`):

`ContextBuilder::build()` produces two hashes from the same inputs:

```
signature_hash = SHA-256(
  system_prompt_hash : response_format_key : tools_hash : tool_choice :
  intent : domain : user_scope : model :
  temp_bucket : top_p_bucket : max_tokens_bucket : seed :
  turn_0:...|turn_1:...|...
)

context_fingerprint = SHA-256(
  system_prompt_hash : response_format_key : tools_hash : tool_choice :
  domain : user_scope : model :
  temp_bucket : top_p_bucket : max_tokens_bucket : seed :
  turn_0:...|turn_1:...|...
)
```

The only difference: `context_fingerprint` omits the query `intent`.

- `signature_hash` → L1 exact-match key (`lc:l1:{sig_hash}`). Two requests must phrase the question identically to share an L1 entry.
- `context_fingerprint` → stored in `CacheEntry::context_signature`; compared by `ValidationService::contextSignatureScore()`. Two differently-phrased questions in the same deployment context share the same fingerprint, allowing L2 semantic hits.

Without the split, max L2 score for a paraphrased query = `0.60 + 0.15 = 0.75`, below the 0.85 threshold. With it, a paraphrase scoring 0.92 cosine gets `0.60×0.92 + 0.25 + 0.15 = 0.952`.

**`CacheDimensions` struct** (`src/orchestrator/ContextBuilder.h`): all LLM call parameters that affect response uniqueness. Passed to `ContextBuilder::build()` from `QueryOrchestrator`.

Framing parameters (exact in key): `system_prompt`, `response_format`, `response_schema`, `tools`, `tool_choice`
Distribution parameters (bucketed): `temperature` (1 d.p.), `top_p` (1 d.p.), `max_tokens` (none/short/medium/long), `seed` (exact or "none"), `model`

**Validation scoring** (`src/validation/ValidationService.cpp`):
```
score = 0.60 × cosine_sim
      + 0.25 × (context_fingerprint == candidate.context_signature ? 1 : 0)
      + 0.15 × (domain match ? 1 : 0)
```
Threshold defaults to 0.85; overridable per-domain via `DOMAIN_THRESHOLDS`. Weights are compile-time constants in `src/validation/ValidationService.h`.

**`CacheEntry` struct** (`src/cache/FaissVectorStore.h`):
- `context_signature` — stores the `context_fingerprint` (not full sig_hash). Used by `ValidationService` for L2 context matching.
- `signature_hash` — stores the full sig_hash. Used by DELETE handlers to reconstruct the correct `lc:l1:{sig_hash}` Redis key for cleanup.

**Two-phase FAISS index** (`src/cache/FaissVectorStore.cpp`):
- Below `MIN_IVF_TRAIN_VEC` (3 900 = 39 × NLIST): `IndexFlatIP` — exact search, 100% recall.
- At/above threshold: auto-migrates to `IndexIVFPQ` — approximate, memory-efficient. Migration trains on all real vectors (no random padding).

**Async write path**: `QueryOrchestrator` returns immediately. All admission + write work happens on the `CacheBuilderWorker` background thread. Write pipeline: `AdmissionController` (frequency gate) → `ResponseQualityFilter` (hard rejects) → `IntelligentAdmissionPolicy` (CVF) → `Templatizer` → `FaissVectorStore::add()` → `Redis MULTI/EXEC (L1 + slots)` → `Redis SADD (domain index)`.

**Tombstone-first DELETE**: `DELETE /cache/{entry_id}` writes tombstone → calls `FaissVectorStore::find()` to get sig_hash + domain → removes from FAISS → deletes `lc:l1:{sig_hash}` → deletes `lc:slots:{domain}:{entry_id}`. Tombstone is checked in `QueryOrchestrator` before returning any L2 hit.

**Static library pattern**: All source files compile into `lettucecache_lib` (STATIC). Both the main executable and test binaries link against it — avoids double compilation.

**Composition root**: `HttpServer` owns all component lifetimes as `unique_ptr`. No other component creates its own dependencies.

**Python sidecar**: FastAPI process serving `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized). `EmbeddingClient` calls it via persistent libcurl handle. Dimension validated on every response against `EMBED_DIM`. 3-state circuit breaker (CLOSED/OPEN/HALF_OPEN, 5-failure threshold, 30s reset).

**TurboQuantizer** (`src/quantization/`): Optional data-oblivious vector quantizer (arXiv:2504.19874). Enabled via `ENABLE_TURBO_QUANT=1`. RHT + Lloyd-Max MSE (3 bits) + QJL residual (1 bit). At 4 bits, d=384: 244 bytes/vector (6.3× vs FP32). `E[TQ_ip(y, encode(x))] = ⟨y, x⟩` — threshold remains valid after compression. Seeds configurable via env vars.

**IVectorStore** (`src/cache/IVectorStore.h`): Abstract interface with `add/search/find/remove/size/persist`. `FaissVectorStore` implements it.

### Configuration

All runtime config is environment variables:

| Var | Default | Notes |
|-----|---------|-------|
| `REDIS_HOST` | `localhost` | |
| `REDIS_PORT` | `6379` | |
| `EMBED_URL` | `http://localhost:8001` | Python sidecar |
| `EMBED_DIM` | `384` | Must match model output dim |
| `OPENAI_API_KEY` | *(empty)* | Empty → stub mode |
| `LLM_MODEL` | `gpt-4o-mini` | Default model name |
| `HTTP_PORT` | `8080` | |
| `FAISS_INDEX_PATH` | `./faiss.index` | Also writes `.meta.json` sidecar |
| `ENABLE_TURBO_QUANT` | *(unset)* | `1` to enable |
| `TURBO_ROTATION_SEED` | `42` | RHT seed |
| `TURBO_QJL_SEED` | `137` | QJL seed |
| `CACHE_QUALITY_THRESHOLD` | `0.40` | Min quality score for admission |
| `DOMAIN_THRESHOLDS` | *(unset)* | JSON e.g. `{"faq":0.75,"compliance":0.92}` |

Tuning parameters requiring source changes:
- Scoring weights: `src/validation/ValidationService.h` (W_COSINE/W_CONTEXT/W_DOMAIN)
- Admission CVF weights: `src/builder/IntelligentAdmissionPolicy.h` (IntelligentAdmissionConfig)
- High-temp bypass threshold: `src/orchestrator/QueryOrchestrator.h` (HIGH_TEMP_THRESHOLD)
- FAISS IVF migration threshold: `src/cache/FaissVectorStore.h` (MIN_IVF_TRAIN_VEC)
- FAISS NLIST/NPROBE/M_PQ: `src/cache/FaissVectorStore.h`

### Redis key schema

| Key pattern | Type | TTL | Contents |
|---|---|---|---|
| `lc:l1:{sig_hash}` | string | 3 600s | Full LLM response |
| `lc:slots:{domain}:{entry_id}` | string | 7 200s | JSON array of slot values |
| `lc:tomb:{entry_id}` | string | 86 400s | Tombstone sentinel ("1") |
| `lc:domain_idx:{domain}` | set | no TTL | Set of entry_ids for the domain |

### Metadata persistence

`FaissVectorStore::persist()` writes two files:
- `<FAISS_INDEX_PATH>` — raw FAISS binary (`faiss::write_index`)
- `<FAISS_INDEX_PATH>.meta.json` — JSON array of `CacheEntry` objects (embeddings, tq_codes_hex, context_signature, signature_hash, domain, template_str, created_at, ivf_trained flag)

On startup, both are loaded. `signature_hash` field is optional in the JSON (empty string for entries written before this field was added — DELETE will not clean up their L1 key, but they expire via TTL).

### Known gaps

- **Redis Streams unused**: `xadd`/`xread` are implemented in `RedisCacheAdapter` but `CacheBuilderWorker` uses an in-process `std::queue` instead. *(still open)*
- **FAISS metadata not persisted**: ~~FIXED~~
- **Redis has no mutex**: ~~FIXED~~
- **Templatizer render dead code**: ~~FIXED~~
- **FAISS exclusive mutex**: ~~FIXED~~
- **Context sort destroys order**: ~~FIXED~~
- **DELETE wrong L1 key**: ~~FIXED~~ — `FaissVectorStore::find()` retrieves sig_hash before removal; correct `lc:l1:{sig_hash}` deleted.
- **L2 context_signature used full sig_hash**: ~~FIXED~~ — `context_fingerprint` (no intent) stored in `CacheEntry::context_signature`; paraphrased queries now get L2 hits.
- **Slot keys not deleted on eviction**: ~~FIXED~~ — both DELETE handlers now clean up slot keys via `find()`.
