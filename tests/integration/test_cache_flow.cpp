/**
 * Integration tests for the full cache flow.
 *
 * Prerequisites (started by docker-compose or CI service containers):
 *   - Redis reachable at REDIS_HOST:REDIS_PORT (default localhost:6379)
 *
 * The embedding sidecar is replaced by a stub LLM + stub embedder so these
 * tests run without network access to OpenAI or the Python sidecar.
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <string>

#include "cache/RedisCacheAdapter.h"
#include "cache/FaissVectorStore.h"
#include "validation/ValidationService.h"
#include "builder/AdmissionController.h"
#include "builder/Templatizer.h"
#include "builder/CacheBuilderWorker.h"

using namespace lettucecache;

namespace {

const char* redisHost() {
    const char* h = std::getenv("REDIS_HOST");
    return h ? h : "localhost";
}

int redisPort() {
    const char* p = std::getenv("REDIS_PORT");
    return p ? std::stoi(p) : 6379;
}

// Minimal fake embedding: fixed 8-dim unit vector
std::vector<float> fakeEmbed(float angle) {
    return {std::cos(angle), std::sin(angle), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

} // anonymous namespace

// ── Redis connectivity ─────────────────────────────────────────────────────

TEST(IntegrationRedis, PingSucceeds) {
    cache::RedisCacheAdapter redis(redisHost(), redisPort());
    EXPECT_TRUE(redis.ping());
}

TEST(IntegrationRedis, SetAndGet) {
    cache::RedisCacheAdapter redis(redisHost(), redisPort());
    const std::string key   = "lc:test:integration:key";
    const std::string value = "hello_lettuce";

    ASSERT_TRUE(redis.set(key, value, 60));
    auto got = redis.get(key);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got.value(), value);
    redis.del(key);
}

TEST(IntegrationRedis, GetMissingKeyReturnsEmpty) {
    cache::RedisCacheAdapter redis(redisHost(), redisPort());
    auto got = redis.get("lc:test:definitely_not_here_xyz");
    EXPECT_FALSE(got.has_value());
}

TEST(IntegrationRedis, DelRemovesKey) {
    cache::RedisCacheAdapter redis(redisHost(), redisPort());
    redis.set("lc:test:del_me", "value", 60);
    EXPECT_TRUE(redis.del("lc:test:del_me"));
    EXPECT_FALSE(redis.get("lc:test:del_me").has_value());
}

// ── FAISS store ───────────────────────────────────────────────────────────

TEST(IntegrationFaiss, AddAndSearchReturnsCandidate) {
    const std::string idx_path = "/tmp/lc_test_faiss.index";
    cache::FaissVectorStore store(8, idx_path);

    cache::CacheEntry entry;
    entry.id                = "e1";
    entry.embedding         = fakeEmbed(0.0f);
    entry.context_signature = "sig1";
    entry.template_str      = "Test answer";
    entry.domain            = "test";

    store.add(entry);
    EXPECT_EQ(store.size(), 1u);

    auto results = store.search(fakeEmbed(0.01f), 3);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].id, "e1");
}

TEST(IntegrationFaiss, SearchOnEmptyReturnsEmpty) {
    cache::FaissVectorStore store(8, "/tmp/lc_test_empty.index");
    auto results = store.search(fakeEmbed(0.0f), 3);
    EXPECT_TRUE(results.empty());
}

TEST(IntegrationFaiss, RemoveEntry) {
    cache::FaissVectorStore store(8, "/tmp/lc_test_remove.index");

    cache::CacheEntry entry;
    entry.id        = "rm1";
    entry.embedding = fakeEmbed(0.5f);
    entry.domain    = "test";
    store.add(entry);
    EXPECT_EQ(store.size(), 1u);

    EXPECT_TRUE(store.remove("rm1"));
    EXPECT_EQ(store.size(), 0u);
}

// ── End-to-end builder worker flow ────────────────────────────────────────

TEST(IntegrationCacheFlow, BuilderWritesToRedisAndFaiss) {
    cache::RedisCacheAdapter redis(redisHost(), redisPort());
    cache::FaissVectorStore  faiss(8, "/tmp/lc_test_builder.index");
    builder::AdmissionController admission(2, 300, 65536);
    builder::Templatizer templatizer;
    builder::CacheBuilderWorker worker(redis, faiss, admission, templatizer);

    worker.start();

    const std::string sig = "integration_test_sig_abc";
    // Simulate 2 queries to meet admission threshold
    admission.recordQuery(sig);
    admission.recordQuery(sig);

    builder::CacheEntryRequest req;
    req.query          = "integration test query";
    req.context        = {};
    req.domain         = "test";
    req.user_id        = "tester";
    req.signature_hash = sig;
    req.llm_response   = "The answer is 42.";
    req.embedding      = fakeEmbed(0.3f);

    worker.enqueue(req);

    // Give the worker time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    worker.stop();

    // L1 should be set
    auto l1 = redis.get("lc:l1:" + sig);
    ASSERT_TRUE(l1.has_value());
    EXPECT_EQ(l1.value(), "The answer is 42.");

    // FAISS should have one entry
    EXPECT_EQ(faiss.size(), 1u);

    // Cleanup
    redis.del("lc:l1:" + sig);
}

TEST(IntegrationCacheFlow, BuilderRejectsWhenAdmissionNotMet) {
    cache::RedisCacheAdapter redis(redisHost(), redisPort());
    cache::FaissVectorStore  faiss(8, "/tmp/lc_test_noadmit.index");
    builder::AdmissionController admission(5, 300, 65536); // high threshold
    builder::Templatizer templatizer;
    builder::CacheBuilderWorker worker(redis, faiss, admission, templatizer);

    worker.start();

    const std::string sig = "noadmit_sig_xyz";
    // Only record once (threshold is 5)
    admission.recordQuery(sig);

    builder::CacheEntryRequest req;
    req.signature_hash = sig;
    req.llm_response   = "response";
    req.embedding      = fakeEmbed(0.1f);
    req.domain         = "test";

    worker.enqueue(req);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    worker.stop();

    EXPECT_FALSE(redis.get("lc:l1:" + sig).has_value());
    EXPECT_EQ(faiss.size(), 0u);
}
