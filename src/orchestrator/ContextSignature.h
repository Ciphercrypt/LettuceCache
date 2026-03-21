#pragma once
#include <string>

namespace lettucecache::orchestrator {

class ContextSignature {
public:
    static std::string sha256(const std::string& input);
    static std::string hashUserId(const std::string& user_id);
};

} // namespace lettucecache::orchestrator
