#include <gtest/gtest.h>
#include "builder/Templatizer.h"

using namespace lettucecache::builder;

TEST(TemplatizerTest, ExtractsNumericSlots) {
    Templatizer t;
    auto result = t.templatize("The price is 42 dollars.");
    EXPECT_NE(result.templ.find("{{SLOT_0}}"), std::string::npos);
    ASSERT_FALSE(result.slot_values.empty());
    EXPECT_EQ(result.slot_values[0], "42");
}

TEST(TemplatizerTest, ExtractsUUIDSlot) {
    Templatizer t;
    std::string resp = "Your order id is 550e8400-e29b-41d4-a716-446655440000 confirmed.";
    auto result = t.templatize(resp);
    EXPECT_NE(result.templ.find("{{SLOT_0}}"), std::string::npos);
    EXPECT_EQ(result.slot_values[0], "550e8400-e29b-41d4-a716-446655440000");
}

TEST(TemplatizerTest, NoHighEntropyTokensNoSlots) {
    Templatizer t;
    auto result = t.templatize("the sky is blue and the grass is green");
    EXPECT_TRUE(result.slot_values.empty());
    EXPECT_EQ(result.templ, "the sky is blue and the grass is green");
}

TEST(TemplatizerTest, RenderFillsSlots) {
    std::string templ = "Hello {{SLOT_0}}, your score is {{SLOT_1}}.";
    std::vector<std::string> vals = {"Alice", "95"};
    std::string rendered = Templatizer::render(templ, vals);
    EXPECT_EQ(rendered, "Hello Alice, your score is 95.");
}

TEST(TemplatizerTest, RenderWithMissingSlotLeavesPlaceholder) {
    std::string templ = "Hello {{SLOT_0}} and {{SLOT_1}}.";
    std::vector<std::string> vals = {"Alice"};
    std::string rendered = Templatizer::render(templ, vals);
    EXPECT_EQ(rendered, "Hello Alice and {{SLOT_1}}.");
}

TEST(TemplatizerTest, MultipleNumbersCreateMultipleSlots) {
    Templatizer t;
    auto result = t.templatize("There are 3 cats and 7 dogs.");
    EXPECT_GE(result.slot_values.size(), 2u);
}

TEST(TemplatizerTest, RoundTripPreservesSemantics) {
    Templatizer t;
    std::string original = "The temperature is 72 degrees.";
    auto result = t.templatize(original);
    std::string restored = Templatizer::render(result.templ, result.slot_values);
    EXPECT_EQ(restored, original);
}
