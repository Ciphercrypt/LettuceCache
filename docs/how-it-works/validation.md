# Validation Scoring

The `ValidationService` decides whether a FAISS candidate is good enough to serve as a cache hit.

## The Score Formula

```python
score = (0.60 * cosine_similarity(query_embedding, candidate_embedding)
       + 0.25 * context_signature_match   # 1.0 if SHA-256 matches, else 0.0
       + 0.15 * domain_match)             # 1.0 if domains match, else 0.0
```

| Component | Weight | Type | Max contribution |
|---|:---:|---|:---:|
| Cosine similarity | 0.60 | Continuous [0, 1] | 0.60 |
| Context signature match | 0.25 | Binary | 0.25 |
| Domain match | 0.15 | Binary | 0.15 |
| **Total** | **1.00** | | **1.00** |

A candidate is returned as a hit when `score ≥ threshold` (default: **0.85**).

## Component Breakdown

### Cosine Similarity (weight: 0.60)

Measures the semantic similarity between the query embedding and the stored embedding:

```
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Computed in pure C++ over the 384-dimensional `all-MiniLM-L6-v2` vectors. A value of 1.0 means the embeddings are identical; 0.0 means orthogonal.

!!! note "FAISS pre-filters"
    FAISS already returns only the top-5 nearest neighbours by L2 distance, so candidates reaching `ValidationService` have reasonable cosine scores. The validation layer adds the context and domain signals on top.

### Context Signature Match (weight: 0.25)

Binary: `1.0` if the query's `signature_hash` equals the candidate's `context_signature`, `0.0` otherwise.

The signature encodes `intent:domain:anon_user_scope`. See [Context Signatures](context-signatures.md) for details.

A mismatch here caps the maximum achievable score at **0.75** — safely below the default threshold.

### Domain Match (weight: 0.15)

Binary: `1.0` if `query.domain == candidate.domain`, `0.0` otherwise.

A domain mismatch (e.g. query is `ecommerce`, candidate is `healthcare`) caps the score at **0.85** — right at the threshold boundary, meaning a high-confidence semantic match in the wrong domain only passes if cosine is near-perfect and context matches exactly.

## Threshold Guidance

| Threshold | Behaviour |
|---|---|
| `0.95` | Very strict. Only near-identical queries with exact context hit. Low hit rate, near-zero false positives. |
| `0.85` *(default)* | Balanced. Allows paraphrase hits within the same context. |
| `0.75` | Relaxed. Higher hit rate; increases risk of semantically-adjacent but wrong answers. |

Change the threshold in `HttpServer.cpp`:

```cpp
validator_ = std::make_unique<validation::ValidationService>(0.85);
```

## Score Examples

```
Query:     "How do I cancel my order?"
Context:   ["I placed an order yesterday", "It hasn't shipped yet"]
Domain:    ecommerce
Signature: a3f2...

Candidate A — "How can I cancel an order?" / same context / ecommerce
  cosine = 0.96, ctx = 1.0, domain = 1.0
  score  = 0.60×0.96 + 0.25×1.0 + 0.15×1.0 = 0.576 + 0.25 + 0.15 = 0.976 → HIT ✅

Candidate B — "How do I cancel my order?" / different context (gym) / fitness
  cosine = 0.99, ctx = 0.0, domain = 0.0
  score  = 0.60×0.99 + 0.25×0.0 + 0.15×0.0 = 0.594 → MISS ❌

Candidate C — "What is the refund process?" / same context / ecommerce
  cosine = 0.78, ctx = 1.0, domain = 1.0
  score  = 0.60×0.78 + 0.25×1.0 + 0.15×1.0 = 0.468 + 0.25 + 0.15 = 0.868 → HIT ✅
```
