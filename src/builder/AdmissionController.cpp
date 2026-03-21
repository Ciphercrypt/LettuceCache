#include "AdmissionController.h"
#include <spdlog/spdlog.h>

namespace lettucecache::builder {

AdmissionController::AdmissionController(int min_frequency,
                                           int window_seconds,
                                           size_t max_response_bytes)
    : min_frequency_(min_frequency),
      window_seconds_(window_seconds),
      max_response_bytes_(max_response_bytes) {}

void AdmissionController::evictExpired() const {
    auto now = std::chrono::steady_clock::now();
    auto it = freq_map_.begin();
    while (it != freq_map_.end()) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - it->second.first_seen).count();
        if (age > window_seconds_) {
            it = freq_map_.erase(it);
        } else {
            ++it;
        }
    }
}

void AdmissionController::recordQuery(const std::string& signature_hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    evictExpired();

    auto& entry = freq_map_[signature_hash];
    if (entry.count == 0) {
        entry.first_seen = std::chrono::steady_clock::now();
    }
    ++entry.count;
}

bool AdmissionController::shouldAdmit(const std::string& signature_hash,
                                        const std::string& response_text) const
{
    // Reject oversized responses
    if (response_text.size() > max_response_bytes_) {
        spdlog::debug("AdmissionController: reject oversized response ({} bytes) for {}",
                      response_text.size(), signature_hash);
        return false;
    }

    // Reject empty responses
    if (response_text.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = freq_map_.find(signature_hash);
    if (it == freq_map_.end()) {
        return false;
    }

    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - it->second.first_seen).count();

    if (age > window_seconds_) {
        return false;
    }

    bool admitted = it->second.count >= min_frequency_;
    if (admitted) {
        spdlog::debug("AdmissionController: admit {} (freq={})",
                      signature_hash, it->second.count);
    }
    return admitted;
}

int AdmissionController::getFrequency(const std::string& signature_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = freq_map_.find(signature_hash);
    return (it != freq_map_.end()) ? it->second.count : 0;
}

} // namespace lettucecache::builder
