#include "HttpServer.h"
#include "../cache/RedisCacheAdapter.h"
#include "../cache/FaissVectorStore.h"
#include "../embedding/EmbeddingClient.h"
#include "../llm/OpenAIAdapter.h"
#include "../validation/ValidationService.h"
#include "../builder/AdmissionController.h"
#include "../builder/IntelligentAdmissionPolicy.h"
#include "../builder/ResponseQualityFilter.h"
#include "../builder/Templatizer.h"
#include "../builder/CacheBuilderWorker.h"
#include "../orchestrator/QueryOrchestrator.h"
#include "../quantization/TurboQuantizer.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <chrono>
#include <thread>

namespace lettucecache::api {

HttpServer::HttpServer(const std::string& redis_host,
                        int redis_port,
                        const std::string& embed_url,
                        const std::string& openai_key,
                        const std::string& faiss_path,
                        int embed_dim,
                        int http_port)
    : http_port_(http_port)
{
    redis_      = std::make_unique<cache::RedisCacheAdapter>(redis_host, redis_port);
    // TurboQuantizer enabled when ENABLE_TURBO_QUANT=1 env var is set.
    // Provides unbiased inner-product estimation + 7.8× embedding compression.
    if (std::getenv("ENABLE_TURBO_QUANT") &&
        std::string(std::getenv("ENABLE_TURBO_QUANT")) == "1")
    {
        const char* rot_env = std::getenv("TURBO_ROTATION_SEED");
        const char* qjl_env = std::getenv("TURBO_QJL_SEED");
        uint64_t rotation_seed = rot_env ? static_cast<uint64_t>(std::stoull(rot_env)) : 42ULL;
        uint64_t qjl_seed      = qjl_env ? static_cast<uint64_t>(std::stoull(qjl_env)) : 137ULL;
        tq_ = std::make_unique<quantization::TurboQuantizer>(
            static_cast<size_t>(embed_dim), rotation_seed, qjl_seed);
        spdlog::info("TurboQuant enabled (dim={} code_size={} bytes rot_seed={} qjl_seed={})",
                     embed_dim, tq_->code_size(), rotation_seed, qjl_seed);
    }
    faiss_      = std::make_unique<cache::FaissVectorStore>(embed_dim, faiss_path,
                                                             tq_.get());
    embedder_   = std::make_unique<embedding::EmbeddingClient>(embed_url, embed_dim);
    const char* model_env = std::getenv("LLM_MODEL");
    default_model_ = model_env ? model_env : "gpt-4o-mini";
    llm_        = std::make_unique<llm::OpenAIAdapter>(openai_key, default_model_);
    spdlog::info("  LLM model: {}", default_model_);
    validator_  = std::make_unique<validation::ValidationService>(0.85, tq_.get());
    admission_      = std::make_unique<builder::AdmissionController>(2, 300, 32768);
    policy_         = std::make_unique<builder::IntelligentAdmissionPolicy>(*faiss_);
    spdlog::info("IntelligentAdmissionPolicy enabled "
                 "(freq×0.30 cost×0.25 quality×0.25 novelty×0.20 threshold=0.42)");
    const char* q_thresh = std::getenv("CACHE_QUALITY_THRESHOLD");
    float quality_thresh = q_thresh ? std::stof(q_thresh) : 0.40f;
    quality_filter_ = std::make_unique<builder::ResponseQualityFilter>(quality_thresh);
    templatizer_    = std::make_unique<builder::Templatizer>();
    builder_        = std::make_unique<builder::CacheBuilderWorker>(
        *redis_, *faiss_, *admission_, *policy_, *quality_filter_, *templatizer_);
    orchestrator_ = std::make_unique<orchestrator::QueryOrchestrator>(
        *redis_, *faiss_, *embedder_, *llm_, *validator_, *builder_, *policy_);
    svr_        = std::make_unique<httplib::Server>();
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::registerRoutes() {
    // ── POST /query ───────────────────────────────────────────────────────
    svr_->Post("/query", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = nlohmann::json::parse(req.body);

            orchestrator::QueryRequest qreq;
            qreq.query          = body.value("query", "");
            qreq.user_id        = body.value("user_id", "anonymous");
            qreq.session_id     = body.value("session_id", "");
            qreq.domain         = body.value("domain", "general");
            qreq.correlation_id = body.value("correlation_id", "");

            // ── LLM framing parameters ─────────────────────────────────────
            qreq.system_prompt   = body.value("system_prompt", "");
            qreq.response_format = body.value("response_format", "text");
            qreq.tool_choice     = body.value("tool_choice", "");
            // response_schema: present only for json_schema format
            if (body.contains("response_schema") &&
                body["response_schema"].is_object()) {
                qreq.response_schema = body["response_schema"].dump();
            }
            // tools: JSON array of tool/function definitions
            if (body.contains("tools") && body["tools"].is_array()) {
                qreq.tools = body["tools"].dump();
            }

            // ── LLM distribution parameters ────────────────────────────────
            qreq.temperature = body.value("temperature", 0.0f);
            qreq.top_p       = body.value("top_p", 1.0f);
            qreq.max_tokens  = body.value("max_tokens", 0);
            // seed: -1 sentinel means "not provided"; treat null same as absent
            if (body.contains("seed") && body["seed"].is_number_integer()) {
                qreq.seed = body["seed"].get<int>();
            }
            qreq.model = body.value("model", default_model_);

            if (body.contains("context") && body["context"].is_array()) {
                for (auto& c : body["context"]) {
                    qreq.context.push_back(c.get<std::string>());
                }
            }

            if (qreq.query.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"query field is required"})", "application/json");
                return;
            }

            auto resp = orchestrator_->process(qreq);

            nlohmann::json out;
            out["answer"]         = resp.answer;
            out["cache_hit"]      = resp.cache_hit;
            out["confidence"]     = resp.confidence;
            out["cache_entry_id"] = resp.cache_entry_id;
            out["latency_ms"]     = resp.latency_ms;

            res.set_content(out.dump(), "application/json");
        } catch (const nlohmann::json::parse_error& e) {
            res.status = 400;
            nlohmann::json err;
            err["error"] = std::string("JSON parse error: ") + e.what();
            res.set_content(err.dump(), "application/json");
        } catch (const std::exception& e) {
            spdlog::error("POST /query exception: {}", e.what());
            res.status = 500;
            nlohmann::json err;
            err["error"] = e.what();
            res.set_content(err.dump(), "application/json");
        }
    });

    // ── GET /health ───────────────────────────────────────────────────────
    svr_->Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        bool redis_ok  = redis_->ping();
        bool embed_ok  = embedder_->healthCheck();

        nlohmann::json out;
        out["status"]          = (redis_ok && embed_ok) ? "ok" : "degraded";
        out["redis"]           = redis_ok;
        out["embedding_sidecar"] = embed_ok;
        out["faiss_entries"]   = static_cast<int64_t>(faiss_->size());
        out["queue_depth"]     = static_cast<int64_t>(builder_->queueDepth());

        res.status = (redis_ok && embed_ok) ? 200 : 503;
        res.set_content(out.dump(), "application/json");
    });

    // ── DELETE /cache/domain/:domain ──────────────────────────────────────
    // Bulk-invalidates all cache entries for a domain. Useful when underlying
    // data changes (e.g. fee schedule update, policy refresh).
    // MUST be registered before /cache/(.+) — cpp-httplib matches in order.
    svr_->Delete(R"(/cache/domain/(.+))",
        [this](const httplib::Request& req, httplib::Response& res) {
        std::string domain = req.matches[1].str();
        std::string set_key = "lc:domain_idx:" + domain;
        auto entry_ids = redis_->smembers(set_key);

        int removed = 0;
        for (const auto& eid : entry_ids) {
            // Tombstone first so concurrent L2 reads skip the entry immediately.
            redis_->setTombstone(eid);

            // Look up entry metadata before removal to get the sig_hash
            // (needed for the L1 key) and domain (needed for the slot key).
            auto entry = faiss_->find(eid);
            if (faiss_->remove(eid)) ++removed;

            if (entry.has_value()) {
                // L1 key is lc:l1:{sig_hash} — NOT lc:l1:{entry_id}
                if (!entry->signature_hash.empty())
                    redis_->del("lc:l1:" + entry->signature_hash);
                // Slot key is lc:slots:{domain}:{entry_id}
                redis_->del("lc:slots:" + entry->domain + ":" + eid);
            }
        }
        redis_->del(set_key);

        nlohmann::json out;
        out["domain"]  = domain;
        out["removed"] = removed;
        res.set_content(out.dump(), "application/json");
    });

    // ── DELETE /cache/:key ────────────────────────────────────────────────
    // Tombstone-first pattern: write tombstone before removing from either
    // store so that any concurrent L2 hit sees the tombstone and skips the
    // entry. If the process crashes between the two remove calls the tombstone
    // prevents the ghost FAISS entry from ever being served.
    svr_->Delete(R"(/cache/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string key = req.matches[1].str();

        // Tombstone first so concurrent L2 reads skip the entry immediately.
        redis_->setTombstone(key);

        // Look up entry metadata before removal to get sig_hash (for L1 key)
        // and domain (for slot key). Both are unavailable after removal.
        auto entry = faiss_->find(key);
        bool faiss_removed = faiss_->remove(key);

        bool l1_removed = false;
        if (entry.has_value()) {
            // L1 key is lc:l1:{sig_hash}, NOT lc:l1:{entry_id}
            if (!entry->signature_hash.empty())
                l1_removed = redis_->del("lc:l1:" + entry->signature_hash);
            // Slot key cleanup — prevents orphaned Redis memory
            redis_->del("lc:slots:" + entry->domain + ":" + key);
        }

        bool ok = faiss_removed || l1_removed;
        nlohmann::json out;
        out["deleted"] = ok;
        out["key"]     = key;
        res.status = ok ? 200 : 404;
        res.set_content(out.dump(), "application/json");
    });

    // ── GET /stats ────────────────────────────────────────────────────────
    svr_->Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json out;
        out["faiss_entries"] = static_cast<int64_t>(faiss_->size());
        out["queue_depth"]   = static_cast<int64_t>(builder_->queueDepth());
        res.set_content(out.dump(), "application/json");
    });

    svr_->set_error_handler([](const httplib::Request&, httplib::Response& res) {
        nlohmann::json err;
        err["error"] = "Not found";
        res.set_content(err.dump(), "application/json");
    });
}

void HttpServer::start() {
    builder_->start();
    registerRoutes();

    spdlog::info("HttpServer listening on port {}", http_port_);
    svr_->listen("0.0.0.0", http_port_);
}

void HttpServer::stop() {
    if (svr_) svr_->stop();
    if (builder_) builder_->stop();
}

} // namespace lettucecache::api
