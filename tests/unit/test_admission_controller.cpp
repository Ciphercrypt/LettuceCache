#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "builder/AdmissionController.h"

using namespace lettucecache::builder;

TEST(AdmissionControllerTest, RejectsBelowFrequency) {
    AdmissionController ac(3, 300, 65536);
    ac.recordQuery("sig1"); // freq = 1
    ac.recordQuery("sig1"); // freq = 2
    EXPECT_FALSE(ac.shouldAdmit("sig1", "some response"));
}

TEST(AdmissionControllerTest, AdmitsAtFrequencyThreshold) {
    AdmissionController ac(2, 300, 65536);
    ac.recordQuery("sig2");
    ac.recordQuery("sig2");
    EXPECT_TRUE(ac.shouldAdmit("sig2", "some response"));
}

TEST(AdmissionControllerTest, RejectsEmptyResponse) {
    AdmissionController ac(1, 300, 65536);
    ac.recordQuery("sig3");
    EXPECT_FALSE(ac.shouldAdmit("sig3", ""));
}

TEST(AdmissionControllerTest, RejectsOversizedResponse) {
    AdmissionController ac(1, 300, 10);  // max 10 bytes
    ac.recordQuery("sig4");
    EXPECT_FALSE(ac.shouldAdmit("sig4", "this response is way too long"));
}

TEST(AdmissionControllerTest, RejectsUnknownSignature) {
    AdmissionController ac(1, 300, 65536);
    EXPECT_FALSE(ac.shouldAdmit("never_seen", "response"));
}

TEST(AdmissionControllerTest, FrequencyCounterIsAccurate) {
    AdmissionController ac(5, 300, 65536);
    for (int i = 0; i < 7; ++i) ac.recordQuery("sig5");
    EXPECT_EQ(ac.getFrequency("sig5"), 7);
}

TEST(AdmissionControllerTest, ReturnsZeroForUnknownSignature) {
    AdmissionController ac(1, 300, 65536);
    EXPECT_EQ(ac.getFrequency("ghost"), 0);
}
