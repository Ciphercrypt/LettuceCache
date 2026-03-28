#pragma once
#include <string>
#include <memory>
#include <atomic>

// Forward declarations to keep the header dependency-light
namespace lettucecache::cache        { class RedisCacheAdapter; class FaissVectorStore; }
namespace lettucecache::embedding    { class EmbeddingClient; }
namespace lettucecache::llm          { class OpenAIAdapter; }
namespace lettucecache::validation   { class ValidationService; }
namespace lettucecache::builder      { class AdmissionController; class ResponseQualityFilter;
                                       class Templatizer; class CacheBuilderWorker; }
namespace lettucecache::orchestrator { class QueryOrchestrator; }
namespace lettucecache::quantization { class TurboQuantizer; }

namespace httplib { class Server; }

namespace lettucecache::api {

class HttpServer {
public:
    HttpServer(const std::string& redis_host,
               int redis_port,
               const std::string& embed_url,
               const std::string& openai_key,
               const std::string& faiss_path,
               int embed_dim,
               int http_port);
    ~HttpServer();

    void start();
    void stop();

private:
    int http_port_;

    // Owned components
    std::unique_ptr<quantization::TurboQuantizer>   tq_;   // nullptr = TQ disabled
    std::unique_ptr<cache::RedisCacheAdapter>        redis_;
    std::unique_ptr<cache::FaissVectorStore>        faiss_;
    std::unique_ptr<embedding::EmbeddingClient>     embedder_;
    std::unique_ptr<llm::OpenAIAdapter>             llm_;
    std::unique_ptr<validation::ValidationService>  validator_;
    std::unique_ptr<builder::AdmissionController>   admission_;
    std::unique_ptr<builder::ResponseQualityFilter> quality_filter_;
    std::unique_ptr<builder::Templatizer>           templatizer_;
    std::unique_ptr<builder::CacheBuilderWorker>    builder_;
    std::unique_ptr<orchestrator::QueryOrchestrator> orchestrator_;
    std::unique_ptr<httplib::Server>                svr_;

    void registerRoutes();
};

} // namespace lettucecache::api
