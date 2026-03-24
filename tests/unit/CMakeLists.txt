cmake_minimum_required(VERSION 3.20)

set(UNIT_SOURCES
    test_validation.cpp
    test_context_signature.cpp
    test_admission_controller.cpp
    test_templatizer.cpp
    test_turboquant.cpp
)

add_executable(unit_tests ${UNIT_SOURCES})

target_link_libraries(unit_tests PRIVATE
    lettucecache_lib
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
