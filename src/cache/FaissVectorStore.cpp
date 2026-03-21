#include "FaissVectorStore.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <chrono>
#include <faiss/impl/FaissException.h>

namespace lettucecache::cache {

FaissVectorStore::FaissVectorStore(int dimension, const std::string& index_path)
    : dim_(dimension), index_path_(index_path)
{
    quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim_);
    index_ = std::make_unique<faiss::IndexIVFPQ>(
        quantizer_.get(), dim_, NLIST, M_PQ, NBITS);
    index_->nprobe = NPROBE;

    std::ifstream test(index_path_);
    if (test.good()) {
        test.close();
        try {
            loadFromDisk();
            spdlog::info("FAISS index loaded from {} entries={}", index_path_, id_to_entry_.size());
        } catch (const std::exception& e) {
            spdlog::warn("Could not load FAISS index ({}); starting fresh.", e.what());
        }
    }
}

FaissVectorStore::~FaissVectorStore() {
    try { persist(); } catch (...) {}
}

void FaissVectorStore::ensureTrained(const std::vector<float>& new_vec) {
    if (trained_) return;

    // Collect existing vectors
    std::vector<float> train_data;
    train_data.reserve((id_to_entry_.size() + 1) * static_cast<size_t>(dim_));
    for (auto& [fid, entry] : id_to_entry_) {
        train_data.insert(train_data.end(),
                          entry.embedding.begin(), entry.embedding.end());
    }
    train_data.insert(train_data.end(), new_vec.begin(), new_vec.end());

    // Pad with cheap pseudo-random noise until MIN_TRAIN_VEC vectors
    size_t current_vecs = train_data.size() / static_cast<size_t>(dim_);
    while (current_vecs < static_cast<size_t>(MIN_TRAIN_VEC)) {
        for (int d = 0; d < dim_; ++d) {
            train_data.push_back(static_cast<float>(std::rand()) /
                                 static_cast<float>(RAND_MAX));
        }
        ++current_vecs;
    }

    index_->train(static_cast<int64_t>(current_vecs), train_data.data());
    trained_ = true;
    spdlog::info("FAISS index trained on {} vectors", current_vecs);
}

void FaissVectorStore::add(const CacheEntry& entry) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (static_cast<int>(entry.embedding.size()) != dim_) {
        spdlog::error("Embedding dim mismatch: expected {} got {}",
                      dim_, entry.embedding.size());
        return;
    }

    if (entry_id_to_faiss_id_.count(entry.id)) {
        spdlog::debug("FAISS dedup: entry {} already indexed", entry.id);
        return;
    }

    ensureTrained(entry.embedding);

    int64_t fid = next_id_++;
    CacheEntry stored = entry;
    stored.faiss_id   = fid;
    stored.created_at = static_cast<long long>(
        std::chrono::system_clock::now().time_since_epoch().count());

    index_->add_with_ids(1, entry.embedding.data(), &fid);
    id_to_entry_[fid]              = std::move(stored);
    entry_id_to_faiss_id_[entry.id] = fid;

    spdlog::debug("FAISS add id={} faiss_id={} total={}", entry.id, fid, next_id_);
}

std::vector<CacheEntry> FaissVectorStore::search(
    const std::vector<float>& query_vec, int top_k)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<CacheEntry> results;

    if (!trained_ || id_to_entry_.empty()) {
        spdlog::debug("FAISS search: index empty or not trained");
        return results;
    }
    if (static_cast<int>(query_vec.size()) != dim_) {
        spdlog::error("Query dim mismatch: expected {} got {}", dim_, query_vec.size());
        return results;
    }

    int k = std::min(top_k, static_cast<int>(id_to_entry_.size()));
    std::vector<faiss::idx_t> labels(k, -1);
    std::vector<float> distances(k, 0.0f);

    try {
        index_->search(1, query_vec.data(), k, distances.data(), labels.data());
    } catch (const faiss::FaissException& e) {
        spdlog::error("FAISS search exception: {}", e.what());
        return results;
    }

    for (int i = 0; i < k; ++i) {
        if (labels[i] < 0) continue;
        auto it = id_to_entry_.find(labels[i]);
        if (it != id_to_entry_.end()) {
            results.push_back(it->second);
        }
    }
    return results;
}

bool FaissVectorStore::remove(const std::string& entry_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entry_id_to_faiss_id_.find(entry_id);
    if (it == entry_id_to_faiss_id_.end()) return false;

    int64_t fid = it->second;
    faiss::IDSelectorBatch selector(1, &fid);
    index_->remove_ids(selector);
    id_to_entry_.erase(fid);
    entry_id_to_faiss_id_.erase(it);
    spdlog::info("FAISS removed entry_id={}", entry_id);
    return true;
}

void FaissVectorStore::persist() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!trained_) return;
    faiss::write_index(index_.get(), index_path_.c_str());
    spdlog::info("FAISS persisted to {}", index_path_);
}

void FaissVectorStore::loadFromDisk() {
    // Called from constructor before mutex is needed
    faiss::Index* raw = faiss::read_index(index_path_.c_str());
    if (!raw) throw std::runtime_error("faiss::read_index returned null");
    auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(raw);
    if (!ivfpq) {
        delete raw;
        throw std::runtime_error("Loaded FAISS index is not IndexIVFPQ");
    }
    index_.reset(ivfpq);
    index_->nprobe = NPROBE;
    trained_ = index_->is_trained;
}

size_t FaissVectorStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return id_to_entry_.size();
}

} // namespace lettucecache::cache
