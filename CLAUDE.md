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
HTTP /query → QueryOrchestrator
    → L1: Redis exact-match on SHA-256 context signature (sub-ms)
    → L2: FAISS IVF+PQ ANN search (top_k=5) + ValidationService composite scoring
    → LLM: OpenAIAdapter fallback (async enqueue to CacheBuilderWorker)
```

### Key architectural constraints

**Two-level cache key design**: `ContextBuilder` extracts intent (first 3 non-stopword tokens), domain, and a hashed user scope, then computes `SHA-256(intent:domain:user_scope)` as `signature_hash`. This is the L1 key and the context component of the L2 validation score. This prevents false hits where identical query text means different things in different conversation contexts.

**Validation scoring** (`ValidationService`): composite score = `0.60 × cosine_sim + 0.25 × context_signature_match + 0.15 × domain_match`. Threshold is 0.85. A perfect cosine + domain match (0.75) still fails if context differs — intentional design. Weights are compile-time constants in `src/validation/ValidationService.h`.

**Static library pattern**: All source files compile into `lettucecache_lib` (STATIC). Both the main executable and test binaries link against this library — avoids double compilation.

**Async write path**: `QueryOrchestrator` returns the LLM response to the caller immediately. Cache population (templatize → Redis SET × 2 → FAISS add) happens on a single background thread in `CacheBuilderWorker` via `std::condition_variable` queue. `AdmissionController` gates writes: a query must appear at least 2× within a 300s window before it's admitted to the cache.

**Composition root**: `HttpServer` owns all component lifetimes as `unique_ptr` and wires them at construction. It is the only place where concrete types are instantiated — everything else works through references/pointers.

**Python sidecar**: Separate FastAPI process serving `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized). The C++ `EmbeddingClient` calls it via libcurl. Both `/embed` (single) and `/embed_batch` (up to 256) endpoints exist; the C++ orchestrator currently only uses `/embed`. EmbeddingClient now has a persistent CURL handle (no per-call TLS handshake) and a 3-state circuit breaker (CLOSED/OPEN/HALF_OPEN, 5-failure threshold, 30s reset).

**TurboQuantizer** (`src/quantization/TurboQuantizer.h/.cpp`): Optional data-oblivious vector quantizer (arXiv:2504.19874). Enabled via `ENABLE_TURBO_QUANT=1` env var. Uses Randomized Hadamard Transform + Lloyd-Max scalar codebooks (1–4 bits) for MSE-optimal compression, plus 1-bit QJL on the residual for unbiased inner-product estimation. At 4 bits, d=384: 196 bytes/vector (7.8× vs FP32). Key property: `E[TQ_ip(y, encode(x))] = <y, x>` — the composite validation threshold remains statistically valid after compression. `IVectorStore` abstract interface added for future Milvus migration.

### Configuration

All runtime config is environment variables. Tuning parameters that require source changes:
- Validation threshold and scoring weights: `src/api/HttpServer.cpp` and `src/validation/ValidationService.h`
- Admission control: `src/api/HttpServer.cpp` (`AdmissionController` constructor)
- FAISS index parameters (NLIST, NPROBE, M_PQ): `src/cache/FaissVectorStore.h`
- TurboQuant enable/disable: `ENABLE_TURBO_QUANT=1` env var (default off)
- TurboQuant rotation/QJL seeds: `TurboQuantizer` constructor (currently hardcoded 42/137)

### Known gaps

- **FAISS metadata not persisted**: ~~FIXED~~ — `id_to_entry_` now serialised to `<index_path>.meta.json` sidecar on every `persist()` call. `loadFromDisk()` loads both the FAISS binary and the JSON sidecar.
- **Redis has no mutex**: `RedisCacheAdapter` uses a single `redisContext*` with no thread protection. cpp-httplib is multi-threaded — concurrent requests race on the same connection. *(still open — connection pool or per-request mutex needed)*
- **Templatizer render path is dead code**: `Templatizer::render()` is implemented and tested but never called in `QueryOrchestrator`. L2 hits return `template_str` raw, which may contain `{{SLOT_N}}` placeholders. *(still open)*
- **Redis Streams unused**: `xadd`/`xread` are implemented in `RedisCacheAdapter` but `CacheBuilderWorker` uses an in-process `std::queue` instead. *(still open)*
- **FAISS read/write uses exclusive mutex**: ~~FIXED~~ — `FaissVectorStore` now uses `std::shared_mutex`; `search()` holds a shared lock (concurrent reads allowed), `add()`/`remove()`/`persist()` hold exclusive locks.
- **Context ordering affects hash**: ~~FIXED~~ — `ContextBuilder` sorts context turns before SHA-256, so identical turns in any order produce the same `signature_hash`.

### New components (added in TurboQuant integration sprint)

**TurboQuantizer** (`src/quantization/`): Data-oblivious vector quantizer (arXiv:2504.19874).
- Enable: `ENABLE_TURBO_QUANT=1`; disable (default): omit env var
- Code layout: `[float32 norm][padded_dim MSE bits][(dim) QJL sign bits]` — d=384 → 244 bytes vs 1536 FP32
- Seeds: rotation seed=42, QJL seed=137 (constructor args)

**IVectorStore** (`src/cache/IVectorStore.h`): Abstract interface with `search/add/remove/size/persist`.
`FaissVectorStore` implements it. Swap for `MilvusVectorStore` (Phase 3) without touching callers.

### Configuration (env vars added in sprint)

| Var | Default | Notes |
|-----|---------|-------|
| `ENABLE_TURBO_QUANT` | *(unset)* | Set to `1` to enable TurboQuantizer for encoding + scoring |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name passed to `OpenAIAdapter` |

### Metadata persistence detail

`FaissVectorStore::persist()` writes two files:
- `<FAISS_INDEX_PATH>` — raw FAISS binary (via `faiss::write_index`)
- `<FAISS_INDEX_PATH>.meta.json` — JSON array of `CacheEntry` objects including embeddings and optional `tq_codes_hex`

On startup, both are loaded. If the `.meta.json` sidecar is missing, the FAISS binary is still loaded (index is usable for training) but all metadata lookups return empty until rebuilt from traffic.
