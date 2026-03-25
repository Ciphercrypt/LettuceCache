#include "ContextBuilder.h"
#include "ContextSignature.h"
#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace lettucecache::orchestrator {

namespace {

// Using unordered_set for O(1) lookups instead of O(N) std::find over vector.
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

// Canonicalize a context array so that identical turns in different orders
// produce the same SHA-256 hash. Sorts lexicographically in-place.
// Fixes gap-l: context array ordering affects signature_hash.
std::vector<std::string> canonicalizeContext(std::vector<std::string> ctx) {
    std::sort(ctx.begin(), ctx.end());
    return ctx;
}

} // anonymous namespace

ContextObject ContextBuilder::build(
    const std::string& query,
    const std::vector<std::string>& context,
    const std::string& domain,
    const std::string& user_id) const
{
    ContextObject obj;
    obj.query         = query;
    // Store the original ordering for display; use canonical ordering for hashing.
    obj.context_turns = context;
    obj.domain        = domain.empty() ? "general" : domain;
    obj.user_scope    = ContextSignature::hashUserId(user_id);
    obj.intent        = extractIntent(query);
    obj.signature_hash = hashSignature(obj.intent, obj.domain, obj.user_scope,
                                        canonicalizeContext(context));
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

std::string ContextBuilder::hashSignature(
    const std::string& intent,
    const std::string& domain,
    const std::string& user_scope,
    const std::vector<std::string>& canonical_context)
{
    // Build a stable, whitespace-free serialisation of the context turns
    // then fold into the hash so context contributes to the signature.
    std::ostringstream ctx_part;
    for (const auto& turn : canonical_context) {
        ctx_part << turn << '|';
    }
    return ContextSignature::sha256(
        intent + ':' + domain + ':' + user_scope + ':' + ctx_part.str());
}

} // namespace lettucecache::orchestrator
