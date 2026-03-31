#include "ContextBuilder.h"
#include "ContextSignature.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace lettucecache::orchestrator {

namespace {

const std::unordered_set<std::string>& stopwords() {
    static const std::unordered_set<std::string> kSet = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "to", "of", "in", "at", "on", "for",
        "with", "by", "from", "up", "about", "into", "through", "what",
        "how", "why", "when", "where", "which", "who", "can", "could",
        "would", "should", "will", "may", "might", "please", "tell", "me"
    };
    return kSet;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

ContextObject ContextBuilder::build(
    const std::string& query,
    const std::vector<std::string>& context,
    const std::string& domain,
    const std::string& user_id,
    const CacheDimensions& dims) const
{
    ContextObject obj;
    obj.query         = query;
    obj.context_turns = context;
    obj.domain        = domain.empty() ? "general" : domain;
    obj.user_scope    = ContextSignature::hashUserId(user_id);
    obj.intent              = extractIntent(query);
    obj.signature_hash      = hashSignature(obj.intent, obj.domain, obj.user_scope,
                                             context, dims);
    obj.context_fingerprint = contextFingerprint(obj.domain, obj.user_scope,
                                                  context, dims);
    return obj;
}

std::string ContextBuilder::extractIntent(const std::string& query) {
    std::string lower = query;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    std::istringstream iss(lower);
    std::string word;
    std::ostringstream intent;
    int count = 0;

    while (iss >> word && count < 3) {
        word.erase(std::remove_if(word.begin(), word.end(),
            [](unsigned char c) { return !std::isalnum(c); }), word.end());
        if (word.empty()) continue;
        if (!stopwords().count(word)) {
            if (count > 0) intent << '_';
            intent << word;
            ++count;
        }
    }
    std::string result = intent.str();
    return result.empty() ? "unknown" : result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Distribution bucketing helpers
// ─────────────────────────────────────────────────────────────────────────────

// Temperature: bucket to 1 d.p.  0.0 and 0.01 → "0.0"; 0.7 and 0.71 → "0.7"
std::string ContextBuilder::bucketTemperature(float t) {
    std::ostringstream oss;
    oss << std::fixed;
    oss.precision(1);
    oss << t;
    return oss.str();
}

// top_p: same 1 d.p. bucketing.  Default 1.0 → "1.0"; 0.95 → "1.0"; 0.1 → "0.1"
std::string ContextBuilder::bucketTopP(float p) {
    std::ostringstream oss;
    oss << std::fixed;
    oss.precision(1);
    oss << p;
    return oss.str();
}

// max_tokens: asymmetric lower-bound buckets.
//   0 (unset)  → "none"  — model decides; treat as effectively unlimited
//   1–200      → "short"
//   201–800    → "medium"
//   >800       → "long"
//
// Asymmetric by design: a cached "long" response is safe to serve for a
// "medium" request (complete content, maybe longer). The reverse is unsafe
// (truncated response served for a long request). The buckets represent the
// *minimum* response length class the caller expects, so cache entries from
// the same bucket are interchangeable but cross-bucket is not.
std::string ContextBuilder::bucketMaxTokens(int n) {
    if (n <= 0)   return "none";
    if (n <= 200) return "short";
    if (n <= 800) return "medium";
    return "long";
}

// seed: exact value when provided (caller wants deterministic output);
// "none" when absent (-1 sentinel) so un-seeded requests share the cache.
std::string ContextBuilder::formatSeed(int seed) {
    if (seed < 0) return "none";
    return std::to_string(seed);
}

// ─────────────────────────────────────────────────────────────────────────────
// Framing helpers
// ─────────────────────────────────────────────────────────────────────────────

// Hash a non-empty field to its first 16 hex chars of SHA-256 (64-bit prefix).
// Returns "none" for empty strings so absent params don't create distinct keys.
std::string ContextBuilder::hashField(const std::string& s) {
    if (s.empty()) return "none";
    return ContextSignature::sha256(s).substr(0, 16);
}

// Produce a canonical response-format key:
//   "text"                        — default prose output
//   "json_object"                 — unstructured JSON
//   "json_schema:{schema_hash}"   — structured JSON, schema hashed so two
//                                   different schemas produce different keys
std::string ContextBuilder::normalizeResponseFormat(const std::string& fmt,
                                                     const std::string& schema) {
    if (fmt.empty() || fmt == "text") return "text";
    if (fmt == "json_object")         return "json_object";
    if (fmt == "json_schema") {
        return "json_schema:" + hashField(schema);
    }
    // Unknown format — treat as opaque key component to be safe
    return fmt;
}

// ─────────────────────────────────────────────────────────────────────────────
// hashSignature — combines all dimensions into a single SHA-256 key
//
// Key structure (ordered for readability; all components separated by ':'):
//
//   [framing namespace]
//   system_prompt_hash  — first 16 hex chars, or "none"
//   response_format_key — "text" | "json_object" | "json_schema:{hash}"
//   tools_hash          — first 16 hex chars of tools JSON, or "none"
//   tool_choice         — exact string, or "auto" default
//
//   [query identity]
//   intent              — first 3 non-stop keywords
//   domain              — application domain
//   user_scope          — anonymised user hash (first 16 hex chars)
//
//   [model + distribution]
//   model               — exact model identifier
//   temp_bucket         — 1 d.p. temperature
//   top_p_bucket        — 1 d.p. top_p
//   max_tokens_bucket   — "none" | "short" | "medium" | "long"
//   seed                — exact int, or "none"
//
//   [conversation]
//   turn_0:...|turn_1:...|...  — positional; order is semantically meaningful
// ─────────────────────────────────────────────────────────────────────────────
std::string ContextBuilder::hashSignature(
    const std::string& intent,
    const std::string& domain,
    const std::string& user_scope,
    const std::vector<std::string>& context,
    const CacheDimensions& dims)
{
    // ── Framing namespace ────────────────────────────────────────────────────
    const std::string sys_hash     = hashField(dims.system_prompt);
    const std::string fmt_key      = normalizeResponseFormat(dims.response_format,
                                                              dims.response_schema);
    const std::string tools_hash   = hashField(dims.tools);
    const std::string tool_choice  = dims.tool_choice.empty() ? "auto" : dims.tool_choice;

    // ── Distribution buckets ─────────────────────────────────────────────────
    const std::string temp_bucket  = bucketTemperature(dims.temperature);
    const std::string top_p_bucket = bucketTopP(dims.top_p);
    const std::string tok_bucket   = bucketMaxTokens(dims.max_tokens);
    const std::string seed_str     = formatSeed(dims.seed);

    // ── Positional context serialisation ─────────────────────────────────────
    std::ostringstream ctx_part;
    for (size_t i = 0; i < context.size(); ++i) {
        ctx_part << "turn_" << i << ':' << context[i] << '|';
    }

    // ── Assemble and hash ─────────────────────────────────────────────────────
    return ContextSignature::sha256(
        sys_hash    + ':' + fmt_key      + ':' +
        tools_hash  + ':' + tool_choice  + ':' +
        intent      + ':' + domain       + ':' + user_scope + ':' +
        dims.model  + ':' + temp_bucket  + ':' + top_p_bucket + ':' +
        tok_bucket  + ':' + seed_str     + ':' +
        ctx_part.str());
}

// ─────────────────────────────────────────────────────────────────────────────
// contextFingerprint — context-only hash for L2 validation
//
// Identical to hashSignature but deliberately omits the query intent.
// Two requests that phrase the same question differently in the same deployment
// context (system_prompt, domain, user_scope, model, format, tools, context
// turns) will produce the same fingerprint.  FAISS cosine similarity handles
// whether the query content is semantically close enough; this fingerprint
// only checks that the deployment context matches so a cached response for
// one phrasing is safe to serve for another.
//
// Without this split the maximum possible L2 validation score for any query
// with a different intent string is 0.75, below the default 0.85 threshold,
// making semantic search irrelevant for paraphrased queries.
// ─────────────────────────────────────────────────────────────────────────────
std::string ContextBuilder::contextFingerprint(
    const std::string& domain,
    const std::string& user_scope,
    const std::vector<std::string>& context,
    const CacheDimensions& dims)
{
    const std::string sys_hash    = hashField(dims.system_prompt);
    const std::string fmt_key     = normalizeResponseFormat(dims.response_format,
                                                             dims.response_schema);
    const std::string tools_hash  = hashField(dims.tools);
    const std::string tool_choice = dims.tool_choice.empty() ? "auto" : dims.tool_choice;
    const std::string temp_bucket = bucketTemperature(dims.temperature);
    const std::string top_p_bucket = bucketTopP(dims.top_p);
    const std::string tok_bucket  = bucketMaxTokens(dims.max_tokens);
    const std::string seed_str    = formatSeed(dims.seed);

    std::ostringstream ctx_part;
    for (size_t i = 0; i < context.size(); ++i) {
        ctx_part << "turn_" << i << ':' << context[i] << '|';
    }

    // intent is intentionally absent — FAISS cosine handles query-level matching
    return ContextSignature::sha256(
        sys_hash   + ':' + fmt_key      + ':' +
        tools_hash + ':' + tool_choice  + ':' +
        domain     + ':' + user_scope   + ':' +
        dims.model + ':' + temp_bucket  + ':' + top_p_bucket + ':' +
        tok_bucket + ':' + seed_str     + ':' +
        ctx_part.str());
}

} // namespace lettucecache::orchestrator
