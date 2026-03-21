#pragma once
#include <string>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace lettucecache::builder {

// Decides whether a query result is worth caching.
// A response is admitted when it has been requested at least
// MIN_FREQUENCY times within WINDOW_SECONDS.
class AdmissionController {
public:
    explicit AdmissionController(int min_frequency = 2,
                                  int window_seconds = 300,
                                  size_t max_response_bytes = 32768);

    bool shouldAdmit(const std::string& signature_hash,
                     const std::string& response_text) const;

    // Called on every query (hit or miss) to update frequency counters.
    void recordQuery(const std::string& signature_hash);

    int getFrequency(const std::string& signature_hash) const;

private:
    int min_frequency_;
    int window_seconds_;
    size_t max_response_bytes_;

    struct Entry {
        int count{0};
        std::chrono::steady_clock::time_point first_seen;
    };

    mutable std::mutex mutex_;
    mutable std::unordered_map<std::string, Entry> freq_map_;

    void evictExpired() const;
};

} // namespace lettucecache::builder
