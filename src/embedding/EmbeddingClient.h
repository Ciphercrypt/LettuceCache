#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct curl_slist;
typedef void CURL;

namespace lettucecache::embedding {

// Three-state circuit breaker for the Python sidecar HTTP calls.
// Prevents the hot path from blocking on a dead sidecar; falls through to LLM
// instead of piling up threads in TCP/TLS timeout waits.
//
// CLOSED → OPEN  : after FAILURE_THRESHOLD consecutive transient failures
// OPEN → HALF_OPEN : after RESET_TIMEOUT_S seconds
// HALF_OPEN → CLOSED : on first success
// HALF_OPEN → OPEN   : on failure
//
// "Transient" = 5xx, conn-refused, timeout. HTTP 4xx is not counted (client error).
enum class CircuitState : uint8_t { CLOSED = 0, OPEN = 1, HALF_OPEN = 2 };

class EmbeddingClient {
public:
    // expected_dim: if > 0, the dimension field in sidecar responses is validated
    // against this value. Mismatch logs an error and returns an empty vector.
    explicit EmbeddingClient(const std::string& base_url, int expected_dim = 0);
    ~EmbeddingClient();

    EmbeddingClient(const EmbeddingClient&)            = delete;
    EmbeddingClient& operator=(const EmbeddingClient&) = delete;

    // Returns empty vector on failure or when circuit is OPEN.
    std::vector<float> embed(const std::string& text);

    // Batch embed — up to 256 texts.
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts);

    bool healthCheck();

    CircuitState circuitState() const { return state_.load(); }

private:
    std::string base_url_;
    int         expected_dim_{0};

    // Persistent CURL handle — reused across calls (no per-call TLS handshake).
    CURL*       curl_{nullptr};
    curl_slist* json_headers_{nullptr};
    mutable std::mutex curl_mutex_;

    // Circuit breaker
    std::atomic<CircuitState> state_{CircuitState::CLOSED};
    int failure_count_{0};
    std::chrono::steady_clock::time_point open_since_;
    mutable std::mutex cb_mutex_;

    static constexpr int FAILURE_THRESHOLD  = 5;
    static constexpr int RESET_TIMEOUT_S    = 30;
    static constexpr int TIMEOUT_MS         = 5000;
    static constexpr int CONNECT_TIMEOUT_MS = 2000;

    // Returns nullopt if circuit OPEN (fail-fast) or on error.
    std::optional<std::string> doPost(const std::string& path,
                                       const std::string& body);

    void recordSuccess();
    void recordFailure();
    bool allowRequest();    // true iff request should proceed

    void initCurl();
    void teardownCurl();
};

} // namespace lettucecache::embedding
