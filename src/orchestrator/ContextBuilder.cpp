#include "ContextBuilder.h"
#include "ContextSignature.h"
#include <algorithm>
#include <sstream>

namespace lettucecache::orchestrator {

namespace {
    static const std::vector<std::string> STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "to", "of", "in", "at", "on", "for",
        "with", "by", "from", "up", "about", "into", "through", "what",
        "how", "why", "when", "where", "which", "who", "can", "could",
        "would", "should", "will", "may", "might", "please", "tell", "me"
    };
} // anonymous namespace

ContextObject ContextBuilder::build(
    const std::string& query,
    const std::vector<std::string>& context,
    const std::string& domain,
    const std::string& user_id) const
{
    ContextObject obj;
    obj.query         = query;
    obj.context_turns = context;
    obj.domain        = domain.empty() ? "general" : domain;
    obj.user_scope    = ContextSignature::hashUserId(user_id);
    obj.intent        = extractIntent(query);
    obj.signature_hash = hashSignature(obj.intent, obj.domain, obj.user_scope);
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
        // Strip non-alphanumeric chars
        word.erase(std::remove_if(word.begin(), word.end(),
            [](unsigned char c) { return !std::isalnum(c); }), word.end());
        if (word.empty()) continue;

        bool is_stop = std::find(STOPWORDS.begin(), STOPWORDS.end(), word) != STOPWORDS.end();
        if (!is_stop) {
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
    const std::string& user_scope)
{
    return ContextSignature::sha256(intent + ':' + domain + ':' + user_scope);
}

} // namespace lettucecache::orchestrator
