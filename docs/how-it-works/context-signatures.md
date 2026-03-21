# Context Signatures

The context signature is the core mechanism that differentiates LettuceCache from a plain semantic cache.

## The Problem It Solves

Two users send the same query:

```
"What is the cancellation policy?"
```

Their conversation context is completely different:

- **User A** has been discussing a hotel reservation
- **User B** has been asking about a software subscription

A cosine similarity on the query alone scores near 1.0. Without context awareness, User B gets the hotel answer — a false hit that's worse than a cache miss.

## How It's Built

The `ContextBuilder` computes a `ContextObject` from every request:

```cpp
struct ContextObject {
    std::string query;
    std::vector<std::string> context_turns;  // prior conversation
    std::string domain;                       // e.g. "ecommerce", "fitness"
    std::string user_scope;                   // anonymised user token
    std::string signature_hash;              // SHA-256(intent:domain:user_scope)
    std::string intent;                       // extracted from query
    std::vector<float> embedding;             // populated after embed call
};
```

### Step 1 — Intent Extraction

The first three non-stopword tokens of the query are extracted as the intent:

```
"What is the cancellation policy for my booking?"
→ stopwords removed: ["cancellation", "policy", "booking"]
→ intent: "cancellation_policy_booking"
```

This is intentionally coarse — it captures the topic without being sensitive to exact wording.

### Step 2 — User Scope Anonymisation

The `user_id` is never stored. It is reduced to a 16-character hex token:

```
SHA-256("user:u_12345").substr(0, 16) → "3a7f2c91b0e4d852"
```

This means two requests from the same user share a cache scope, but the actual identity is not recoverable from the stored data.

### Step 3 — Signature Hash

The three components are joined and hashed:

```
signature = SHA-256("cancellation_policy_booking:ecommerce:3a7f2c91b0e4d852")
          = "a3f2c1..."
```

This is the L1 Redis key suffix and the `context_signature` field stored on each FAISS entry.

## Signature Matching in Validation

In `ValidationService`, the context score component is binary:

```cpp
double contextSignatureScore(...) {
    return (query_ctx.signature_hash == candidate.context_signature) ? 1.0 : 0.0;
}
```

Combined with the 0.25 weight, a signature mismatch reduces the maximum achievable score to **0.75** — below the default threshold of **0.85**. The cache will never serve a hit across different context signatures, regardless of how semantically similar the queries are.

## Example

| Scenario | Cosine | Ctx sig | Domain | Score | Result |
|---|:---:|:---:|:---:|:---:|:---:|
| Same query, same context, same domain | 0.97 | 1.0 | 1.0 | **0.97** | HIT |
| Same query, different context | 0.97 | 0.0 | 1.0 | **0.73** | MISS |
| Same query, same context, different domain | 0.97 | 0.0 | 0.0 | **0.58** | MISS |
| Similar query, same context | 0.85 | 1.0 | 1.0 | **0.76** | MISS (below threshold) |
| Very similar query, same context | 0.93 | 1.0 | 1.0 | **0.96** | HIT |
