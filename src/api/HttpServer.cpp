#include "HttpServer.h"
#include "../cache/RedisCacheAdapter.h"
#include "../cache/FaissVectorStore.h"
#include "../embedding/EmbeddingClient.h"
#include "../llm/OpenAIAdapter.h"
#include "../validation/ValidationService.h"
#include "../builder/AdmissionController.h"
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
        tq_ = std::make_unique<quantization::TurboQuantizer>(
            static_cast<size_t>(embed_dim));
        spdlog::info("TurboQuant enabled (dim={} code_size={} bytes)",
                     embed_dim, tq_->code_size());
    }
    faiss_      = std::make_unique<cache::FaissVectorStore>(embed_dim, faiss_path,
                                                             tq_.get());
    embedder_   = std::make_unique<embedding::EmbeddingClient>(embed_url);
    const char* model_env = std::getenv("LLM_MODEL");
    std::string llm_model = model_env ? model_env : "gpt-4o-mini";
    llm_        = std::make_unique<llm::OpenAIAdapter>(openai_key, llm_model);
    spdlog::info("  LLM model: {}", llm_model);
    validator_  = std::make_unique<validation::ValidationService>(0.85, tq_.get());
    admission_      = std::make_unique<builder::AdmissionController>(2, 300, 32768);
    // ResponseQualityFilter threshold configurable via CACHE_QUALITY_THRESHOLD env var
    const char* q_thresh = std::getenv("CACHE_QUALITY_THRESHOLD");
    float quality_thresh = q_thresh ? std::stof(q_thresh) : 0.40f;
    quality_filter_ = std::make_unique<builder::ResponseQualityFilter>(quality_thresh);
    spdlog::info("ResponseQualityFilter enabled (threshold={:.2f})", quality_thresh);
    templatizer_    = std::make_unique<builder::Templatizer>();
    builder_        = std::make_unique<builder::CacheBuilderWorker>(
        *redis_, *faiss_, *admission_, *quality_filter_, *templatizer_);
    orchestrator_ = std::make_unique<orchestrator::QueryOrchestrator>(
        *redis_, *faiss_, *embedder_, *llm_, *validator_, *builder_);
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

    // ── DELETE /cache/:key ────────────────────────────────────────────────
    svr_->Delete(R"(/cache/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        std::string key = req.matches[1].str();
        bool ok = faiss_->remove(key) || redis_->del("lc:l1:" + key);
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
