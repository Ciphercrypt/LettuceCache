#pragma once
#include <string>
#include <vector>

namespace lettucecache::orchestrator {

struct ContextObject {
    std::string query;
    std::vector<std::string> context_turns;
    std::string domain;
    std::string user_scope;
    std::string signature_hash;  // SHA-256(intent:domain:user_scope:sorted_context)
    std::string intent;          // first 3 non-stop keywords, lowercase
    std::vector<float> embedding; // populated by orchestrator after embed call
};

class ContextBuilder {
public:
    ContextObject build(
        const std::string& query,
        const std::vector<std::string>& context,
        const std::string& domain,
        const std::string& user_id
    ) const;

    static std::string extractIntent(const std::string& query);

private:
    static std::string hashSignature(const std::string& intent,
                                      const std::string& domain,
                                      const std::string& user_scope,
                                      const std::vector<std::string>& canonical_context);
};

} // namespace lettucecache::orchestrator
