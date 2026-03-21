#include <gtest/gtest.h>
#include "orchestrator/ContextSignature.h"
#include "orchestrator/ContextBuilder.h"

using namespace lettucecache::orchestrator;

// ── ContextSignature ──────────────────────────────────────────────────────

TEST(ContextSignatureTest, Sha256KnownVector) {
    // echo -n "" | sha256sum => e3b0c44298fc1c149afbf4c8996fb924...
    std::string empty_hash = ContextSignature::sha256("");
    EXPECT_EQ(empty_hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(ContextSignatureTest, Sha256NonEmpty) {
    std::string h = ContextSignature::sha256("hello");
    EXPECT_EQ(h.size(), 64u);
    // Must be hex
    for (char c : h) {
        EXPECT_TRUE(std::isxdigit(static_cast<unsigned char>(c)));
    }
}

TEST(ContextSignatureTest, HashUserIdIsDeterministic) {
    std::string h1 = ContextSignature::hashUserId("alice");
    std::string h2 = ContextSignature::hashUserId("alice");
    EXPECT_EQ(h1, h2);
    EXPECT_EQ(h1.size(), 16u);
}

TEST(ContextSignatureTest, DifferentUsersGiveDifferentHashes) {
    EXPECT_NE(ContextSignature::hashUserId("alice"),
              ContextSignature::hashUserId("bob"));
}

// ── ContextBuilder ────────────────────────────────────────────────────────

TEST(ContextBuilderTest, BuildPopulatesAllFields) {
    ContextBuilder builder;
    auto obj = builder.build("What is the capital of France?", {}, "geography", "user1");

    EXPECT_EQ(obj.query, "What is the capital of France?");
    EXPECT_EQ(obj.domain, "geography");
    EXPECT_FALSE(obj.intent.empty());
    EXPECT_EQ(obj.signature_hash.size(), 64u);
    EXPECT_FALSE(obj.user_scope.empty());
}

TEST(ContextBuilderTest, EmptyDomainDefaultsToGeneral) {
    ContextBuilder builder;
    auto obj = builder.build("hello", {}, "", "u");
    EXPECT_EQ(obj.domain, "general");
}

TEST(ContextBuilderTest, SameDomainAndUserGiveSameSignature) {
    ContextBuilder builder;
    auto o1 = builder.build("capital of france", {}, "geo", "user1");
    auto o2 = builder.build("capital of france", {}, "geo", "user1");
    EXPECT_EQ(o1.signature_hash, o2.signature_hash);
}

TEST(ContextBuilderTest, DifferentDomainGivesDifferentSignature) {
    ContextBuilder builder;
    auto o1 = builder.build("capital of france", {}, "geo", "user1");
    auto o2 = builder.build("capital of france", {}, "history", "user1");
    EXPECT_NE(o1.signature_hash, o2.signature_hash);
}

TEST(ContextBuilderTest, IntentExtractorSkipsStopwords) {
    std::string intent = ContextBuilder::extractIntent("What is the capital of France");
    // "What", "is", "the", "of" are stopwords; "capital" and "France" are kept
    EXPECT_NE(intent.find("capital"), std::string::npos);
}

TEST(ContextBuilderTest, IntentExtractorHandlesEmptyQuery) {
    std::string intent = ContextBuilder::extractIntent("");
    EXPECT_EQ(intent, "unknown");
}
