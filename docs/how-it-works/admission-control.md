# Admission Control

Not every LLM response should be cached. `AdmissionController` acts as a gatekeeper before any entry enters the FAISS index.

## Why It Matters

Without admission control:

- **One-off queries** pollute the index, reducing search precision
- **Ephemeral responses** (e.g. "your order #12345 ships tomorrow") get cached and served incorrectly
- The FAISS index grows unboundedly, degrading search latency

## The Rules

An entry is admitted to the cache only when **all** of the following are true:

| Rule | Default | Description |
|---|---|---|
| Minimum frequency | 2 | The same context signature must have been seen at least N times |
| Time window | 300s (5 min) | Within a rolling window — resets if no activity |
| Max response size | 32 KB | Responses larger than this are rejected (likely too specific to cache usefully) |

## Frequency Counting

`AdmissionController` maintains an in-memory map of `{signature_hash → {count, first_seen}}`:

```
Request 1: signature=a3f2... → count=1, first_seen=T
Request 2: signature=a3f2... → count=2 → shouldAdmit() = true ✅
Request 3: signature=a3f2... → count=3 → shouldAdmit() = true ✅

After 5 minutes with no activity: entry expires → count resets
Request 4: signature=a3f2... → count=1 → shouldAdmit() = false ❌
```

`recordQuery()` is called on every request (hit or miss) to keep counts current. `evictExpired()` runs lazily on each check to clean up stale entries.

## Configuration

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

## Interaction with Templatizer

Even when `AdmissionController` admits an entry, the `Templatizer` strips high-entropy tokens before storage. This means the frequency gate and the template extraction work together:

1. Frequency gate ensures only repeated query patterns are cached
2. Templatizer ensures the stored response is generalisable, not specific to one instance

See [Async Write Path](async-write-path.md) for the full pipeline.
