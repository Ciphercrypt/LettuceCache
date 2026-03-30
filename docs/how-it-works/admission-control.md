# Admission Control

Not every LLM response should be cached. Two independent gatekeepers run in the `CacheBuilderWorker` pipeline before any entry enters the FAISS index: a **frequency gate** (`AdmissionController`) and a **value gate** (`IntelligentAdmissionPolicy`).

## Why It Matters

Without admission control:

- **One-off queries** pollute the index, reducing search precision
- **Ephemeral responses** (e.g. "your order #12345 ships tomorrow") get cached and served incorrectly
- The FAISS index grows unboundedly, degrading search latency

---

## Stage 1 — Frequency Gate (`AdmissionController`)

An entry passes the frequency gate only when **all** of the following are true:

| Rule | Default | Description |
|---|---|---|
| Minimum frequency | 2 | The same context signature must have been seen at least N times |
| Time window | 300s (5 min) | Within a rolling window — resets if no activity |
| Max response size | 32 KB | Responses larger than this are rejected (likely too specific to cache usefully) |

### Frequency Counting

`AdmissionController` maintains an in-memory map of `{signature_hash → {count, first_seen}}`:

```
Request 1: signature=a3f2... → count=1, first_seen=T
Request 2: signature=a3f2... → count=2 → shouldAdmit() = true ✅
Request 3: signature=a3f2... → count=3 → shouldAdmit() = true ✅

After 5 minutes with no activity: entry expires → count resets
Request 4: signature=a3f2... → count=1 → shouldAdmit() = false ❌
```

`recordQuery()` is called on every request (hit or miss) to keep counts current. `evictExpired()` runs lazily on each check to clean up stale entries.

### Configuration

```cpp
admission_ = std::make_unique<builder::AdmissionController>(
    2,       // min_frequency
    300,     // window_seconds
    32768    // max_response_bytes
);
```

Adjust these based on your traffic patterns:

- **High-traffic production**: keep defaults or lower `min_frequency` to `1`
- **Low-traffic / exploratory**: raise `min_frequency` to `3–5` to only cache well-established patterns
- **Batch/scheduled queries**: consider raising `window_seconds` to `3600` (1 hour)

---

## Stage 2 — Value Gate (`IntelligentAdmissionPolicy`)

Entries that pass the frequency gate are scored by a multi-signal **Cache Value Function (CVF)**:

```
CVF = 0.30 × frequency_score      (exponential-decay recency weighting)
    + 0.25 × generation_cost       (token_count × model_tier multiplier)
    + 0.25 × quality_score         (from ResponseQualityFilter)
    + 0.20 × novelty_score         (MMR: 1 − max_cosine_to_existing_cache)
```

An entry is only written to FAISS if `CVF ≥ threshold` (per-domain, adaptive).

### CVF Component Details

| Component | Signal | Rationale |
|---|---|---|
| `frequency_score` | Exponential decay over recency-weighted hit count | Rewards patterns that recur, discounts one-off spikes |
| `generation_cost` | `token_count × model_tier_multiplier` | Expensive LLM calls (GPT-4o, long outputs) are worth caching more |
| `quality_score` | Output of `ResponseQualityFilter` | Rejects conversational, session-bound, refusal, and time-sensitive answers |
| `novelty_score` | `1 − max_cosine_similarity_to_existing_cache` | Prevents near-duplicate entries from polluting the index |

### Hard Rejection Rules

| Rule | Threshold | Effect |
|---|---|---|
| Near-duplicate | cosine > 0.94 to any existing entry | Hard rejected regardless of CVF |
| Low quality | `quality_score < CACHE_QUALITY_THRESHOLD` (default 0.40) | Hard rejected before CVF calculation |

### Adaptive Threshold

Each domain maintains its own admission threshold, adjusted based on observed hit rates:

- Domains with hit rate > 30% get a **relaxed threshold** (more entries admitted)
- Domains with low hit rate keep the default threshold

### Per-request Admission Summary

| Call # | `AdmissionController` | `IntelligentAdmissionPolicy` | Cached? |
|---|---|---|---|
| 1st miss | count=1, below min_frequency | — (not reached) | No |
| 2nd miss | count=2, admitted | CVF evaluated | Only if CVF ≥ threshold |
| 3rd+ | count≥2, admitted | CVF evaluated | Yes (if CVF passes) |

This means the **earliest a query can be cached is after the 2nd LLM call** for that context signature.

---

## Interaction with Templatizer

Even when `AdmissionController` admits an entry, the `Templatizer` strips high-entropy tokens before storage. This means the frequency gate and the template extraction work together:

1. Frequency gate ensures only repeated query patterns are cached
2. Templatizer ensures the stored response is generalisable, not specific to one instance

See [Async Write Path](async-write-path.md) for the full pipeline.
