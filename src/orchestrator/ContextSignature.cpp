#include "ContextSignature.h"
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

namespace lettucecache::orchestrator {

std::string ContextSignature::sha256(const std::string& input) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(input.data()),
           input.size(), hash);

    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(hash[i]);
    }
    return oss.str();
}

std::string ContextSignature::hashUserId(const std::string& user_id) {
    // First 16 hex chars of SHA-256 give a 64-bit anonymous scope token.
    return sha256("user:" + user_id).substr(0, 16);
}

} // namespace lettucecache::orchestrator
