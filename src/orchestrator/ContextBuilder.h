#pragma once
#include <string>
#include <vector>

namespace lettucecache::orchestrator {

struct ContextObject {
    std::string query;
    std::vector<std::string> context_turns;
    std::string domain;
    std::string user_scope;
    std::string intent;           // first 3 non-stop keywords, lowercase
    std::vector<float> embedding; // populated by orchestrator after embed call

    // ── Two separate hashes serve two distinct roles ───────────────────────
    //
    // signature_hash — includes query intent + all dims → used as L1 exact-match key.
    //   Two requests must agree on every dimension (including how the query is
    //   phrased) to share an L1 entry.
    //
    // context_fingerprint — excludes query intent → used for L2 context validation.
    //   Two requests with different phrasings of the same question in the same
    //   deployment context (same system_prompt, domain, user, model, format, tools,
    //   conversation turns) will share this fingerprint.  FAISS cosine similarity
    //   handles the query-level semantic matching; the fingerprint only verifies
    //   that the deployment context is compatible.
    //
    // Without this split: max L2 score for a paraphrased query = 0.75 (below 0.85
    // threshold), making semantic search useless for any query using different words.
    std::string signature_hash;       // SHA-256(intent + all dims) — L1 key
    std::string context_fingerprint;  // SHA-256(dims excluding intent) — L2 context check
};

// All LLM call parameters that affect response uniqueness and therefore
// must be reflected in the cache key.
//
// Split into two categories (see research/cache-key-dimensions.md):
//
//  Framing parameters — change *what kind of response is possible*.
//    Omitting any one allows structurally incompatible responses to collide.
//    Included exactly (or as a short hash):
//      system_prompt, response_format, response_schema, tools, tool_choice
//
//  Distribution parameters — shift the probability distribution over the
//    same response space.  Bucketed so adjacent values don't shard the cache:
//      temperature (1 d.p.), top_p (1 d.p.), max_tokens (3 size buckets), seed
struct CacheDimensions {
    // ── Framing ────────────────────────────────────────────────────────────
    std::string system_prompt;        // static deployment instructions (not per-user)
    std::string response_format;      // "text" | "json_object" | "json_schema"
    std::string response_schema;      // JSON schema string; only for "json_schema" mode
    std::string tools;                // JSON array of tool definitions; empty = none
    std::string tool_choice;          // "auto" | "none" | "required" | function name

    // ── Distribution ───────────────────────────────────────────────────────
    float       temperature{0.0f};    // bucketed to 1 decimal place
    float       top_p{1.0f};          // bucketed to 1 decimal place; default = full dist
    int         max_tokens{0};        // 0 = model decides; bucketed: short/medium/long/none
    int         seed{-1};             // -1 = not provided; else exact value in key
    std::string model;                // exact model identifier
};

class ContextBuilder {
public:
    ContextObject build(
        const std::string& query,
        const std::vector<std::string>& context,
        const std::string& domain,
        const std::string& user_id,
        const CacheDimensions& dims = {}
    ) const;

    static std::string extractIntent(const std::string& query);

private:
    static std::string hashSignature(
        const std::string& intent,
        const std::string& domain,
        const std::string& user_scope,
        const std::vector<std::string>& context,
        const CacheDimensions& dims);

    // Context-only fingerprint — excludes intent so paraphrased queries in the
    // same deployment context share the same fingerprint for L2 validation.
    static std::string contextFingerprint(
        const std::string& domain,
        const std::string& user_scope,
        const std::vector<std::string>& context,
        const CacheDimensions& dims);

    // Bucketing helpers — keep adjacent values in the same cache partition
    static std::string bucketTemperature(float t);
    static std::string bucketTopP(float p);
    static std::string bucketMaxTokens(int n);
    static std::string formatSeed(int seed);

    // Framing helpers — short SHA-256 prefix, or "none" for empty/absent values
    static std::string hashField(const std::string& s);
    static std::string normalizeResponseFormat(const std::string& fmt,
                                               const std::string& schema);
};

} // namespace lettucecache::orchestrator
