cmake_minimum_required(VERSION 3.20)

add_executable(integration_tests test_cache_flow.cpp)

target_link_libraries(integration_tests PRIVATE
    lettucecache_lib
    GTest::gtest
    GTest::gtest_main
)

# Integration tests require live Redis — skip in CI unless INTEGRATION_TESTS=1
if(NOT DEFINED ENV{INTEGRATION_TESTS})
    message(STATUS "Skipping integration test registration (set INTEGRATION_TESTS=1 to enable)")
else()
    include(GoogleTest)
    gtest_discover_tests(integration_tests)
endif()
