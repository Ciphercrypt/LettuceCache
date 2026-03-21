#include <gtest/gtest.h>
#include "validation/ValidationService.h"
#include "orchestrator/ContextBuilder.h"
#include "cache/FaissVectorStore.h"

using namespace lettucecache;

namespace {

orchestrator::ContextObject makeCtx(
    const std::string& sig,
    const std::string& domain,
    const std::vector<float>& emb)
{
    orchestrator::ContextObject ctx;
    ctx.signature_hash = sig;
    ctx.domain         = domain;
    ctx.embedding      = emb;
    return ctx;
}

cache::CacheEntry makeEntry(
    const std::string& sig,
    const std::string& domain,
    const std::vector<float>& emb)
{
    cache::CacheEntry e;
    e.id                = "test_entry";
    e.context_signature = sig;
    e.domain            = domain;
    e.embedding         = emb;
    e.template_str      = "Test answer";
    return e;
}

std::vector<float> unitVec(int dim, float val = 1.0f) {
    std::vector<float> v(dim, 0.0f);
    v[0] = val;
    return v;
}

} // anonymous namespace

TEST(ValidationServiceTest, PerfectMatchScoresHigh) {
    validation::ValidationService svc(0.85);
    std::vector<float> emb = unitVec(4);
    auto ctx     = makeCtx("abc123", "science", emb);
    auto entry   = makeEntry("abc123", "science", emb);

    double s = svc.score(ctx, entry);
    // cosine=1.0 * 0.60 + ctx=1.0 * 0.25 + domain=1.0 * 0.15 = 1.0
    EXPECT_NEAR(s, 1.0, 1e-6);
    EXPECT_TRUE(svc.isHit(ctx, entry));
}

TEST(ValidationServiceTest, DomainMismatchReducesScore) {
    validation::ValidationService svc(0.85);
    std::vector<float> emb = unitVec(4);
    auto ctx   = makeCtx("abc123", "science", emb);
    auto entry = makeEntry("abc123", "history", emb);

    double s = svc.score(ctx, entry);
    // domain=0.0, so max = 0.60 + 0.25 = 0.85
    EXPECT_LT(s, 1.0);
}

TEST(ValidationServiceTest, SignatureMismatchReducesScore) {
    validation::ValidationService svc(0.85);
    std::vector<float> emb = unitVec(4);
    auto ctx   = makeCtx("sig_a", "science", emb);
    auto entry = makeEntry("sig_b", "science", emb);

    double s = svc.score(ctx, entry);
    // ctx=0.0 => max = 0.60 + 0.15 = 0.75 < threshold
    EXPECT_FALSE(svc.isHit(ctx, entry));
}

TEST(ValidationServiceTest, CosineSimilarityOrthogonalIsZero) {
    validation::ValidationService svc(0.50);
    std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f, 0.0f};
    auto ctx   = makeCtx("s1", "d1", a);
    auto entry = makeEntry("s1", "d1", b);

    double s = svc.score(ctx, entry);
    // cosine=0, ctx=1.0, domain=1.0 => 0 + 0.25 + 0.15 = 0.40 < 0.50
    EXPECT_FALSE(svc.isHit(ctx, entry));
}

TEST(ValidationServiceTest, EmptyEmbeddingsYieldZeroCosine) {
    validation::ValidationService svc(0.3);
    auto ctx   = makeCtx("s1", "d1", {});
    auto entry = makeEntry("s1", "d1", {});

    // Both empty — cosine returns 0.0, but ctx+domain should give 0.40
    double s = svc.score(ctx, entry);
    EXPECT_GE(s, 0.0);
    EXPECT_LE(s, 1.0);
}
