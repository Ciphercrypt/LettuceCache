#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include "api/HttpServer.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

static std::atomic<bool> g_running{true};

static void signalHandler(int sig) {
    spdlog::info("Received signal {} — initiating shutdown", sig);
    g_running.store(false);
}

static const char* envOrDefault(const char* name, const char* fallback) {
    const char* val = std::getenv(name);
    return val ? val : fallback;
}

static int envInt(const char* name, int fallback) {
    const char* val = std::getenv(name);
    if (!val) return fallback;
    try { return std::stoi(val); } catch (...) { return fallback; }
}

int main(int /*argc*/, char* /*argv*/[]) {
    auto console = spdlog::stdout_color_mt("lettucecache");
    spdlog::set_default_logger(console);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%dT%H:%M:%S.%e] [%-5l] [%n] %v");

    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    const char* redis_host  = envOrDefault("REDIS_HOST",        "localhost");
    const int   redis_port  = envInt      ("REDIS_PORT",        6379);
    const char* embed_url   = envOrDefault("EMBED_URL",         "http://localhost:8001");
    const char* openai_key  = envOrDefault("OPENAI_API_KEY",    "");
    const int   http_port   = envInt      ("HTTP_PORT",         8080);
    const char* faiss_path  = envOrDefault("FAISS_INDEX_PATH",  "./faiss.index");
    const int   embed_dim   = envInt      ("EMBED_DIM",         384);

    spdlog::info("LettuceCache v1.0.0 starting");
    spdlog::info("  Redis    : {}:{}", redis_host, redis_port);
    spdlog::info("  EmbedURL : {}", embed_url);
    spdlog::info("  HTTP port: {}", http_port);
    spdlog::info("  FAISS    : {} (dim={})", faiss_path, embed_dim);
    spdlog::info("  LLM      : {}", std::string(openai_key).empty() ? "disabled" : "openai");

    try {
        lettucecache::api::HttpServer server(
            redis_host, redis_port,
            embed_url, openai_key,
            faiss_path, embed_dim,
            http_port
        );

        // Run server on a background thread so we can react to signals on main.
        std::thread server_thread([&server] { server.start(); });

        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        server.stop();
        if (server_thread.joinable()) server_thread.join();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }

    spdlog::info("LettuceCache stopped cleanly.");
    return 0;
}
