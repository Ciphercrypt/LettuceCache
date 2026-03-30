# Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and edit.

## Orchestrator

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `EMBED_URL` | `http://localhost:8001` | Python sidecar base URL |
| `EMBED_DIM` | `384` | Embedding dimension — must match the model |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key. If blank, the LLM adapter returns a stub response (useful for testing) |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model name passed to the LLM adapter |
| `HTTP_PORT` | `8080` | Port the C++ server listens on |
| `FAISS_INDEX_PATH` | `./faiss.index` | Path to persist and load the FAISS index on disk |
| `ENABLE_TURBO_QUANT` | *(unset)* | Set to `1` to enable TurboQuantizer compression (7.8× smaller vectors, zero bias on cosine estimates) |
| `CACHE_QUALITY_THRESHOLD` | `0.40` | Minimum `ResponseQualityFilter` score for a response to be admitted to the cache |

## Python Sidecar

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Any `sentence-transformers` compatible model name |
| `WORKERS` | `2` | Number of uvicorn worker processes |

## Tuning

The following are compile-time or constructor defaults. They can be changed in the source if you need different behaviour.

### Validation Threshold

In `src/api/HttpServer.cpp`:

```cpp
validator_ = std::make_unique<validation::ValidationService>(0.85);
//                                                            ^^^^ default threshold
```

Lower values (e.g. `0.75`) increase hit rate but risk false positives.
Higher values (e.g. `0.95`) reduce false positives but lower hit rate.

### Admission Control

```cpp
admission_ = std::make_unique<builder::AdmissionController>(
    2,      // min_frequency: seen at least N times before caching
    300,    // window_seconds: within this rolling window
    32768   // max_response_bytes: reject responses larger than this
);
```

### FAISS Index Parameters

In `src/cache/FaissVectorStore.h`:

```cpp
static constexpr int NLIST  = 100;   // IVF: number of Voronoi cells
static constexpr int M_PQ   = 8;     // PQ: number of sub-quantizers
static constexpr int NBITS  = 8;     // PQ: bits per sub-quantizer
static constexpr int NPROBE = 10;    // search: cells to probe at query time
```

More `NPROBE` = higher recall, higher latency. Tune based on your index size and p95 target.

## Scoring Weights

In `src/validation/ValidationService.h`:

```cpp
static constexpr double W_COSINE  = 0.60;
static constexpr double W_CONTEXT = 0.25;
static constexpr double W_DOMAIN  = 0.15;
```

The composite score is:

```
score = 0.60 × cosine_similarity
      + 0.25 × context_signature_match
      + 0.15 × domain_match
```

Increasing `W_CONTEXT` makes the cache stricter about context boundaries. Increasing `W_COSINE` makes it more meaning-based.
