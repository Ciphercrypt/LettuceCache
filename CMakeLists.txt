cmake_minimum_required(VERSION 3.20)
project(LettuceCache VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)

# hiredis
pkg_check_modules(HIREDIS REQUIRED hiredis)

# FAISS
find_package(faiss QUIET)
if(NOT faiss_FOUND)
    pkg_check_modules(FAISS REQUIRED faiss)
endif()

# libcurl
find_package(CURL REQUIRED)

# OpenSSL (SHA-256)
find_package(OpenSSL REQUIRED)

# nlohmann/json (header-only, FetchContent fallback)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# cpp-httplib (header-only)
include(FetchContent)
FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.tar.gz
)
FetchContent_MakeAvailable(httplib)

# spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.tar.gz
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# GTest for unit/integration tests
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Library target (shared between main + tests) ──────────────────────────
set(LIB_SOURCES
    src/orchestrator/QueryOrchestrator.cpp
    src/orchestrator/ContextBuilder.cpp
    src/orchestrator/ContextSignature.cpp
    src/cache/RedisCacheAdapter.cpp
    src/cache/FaissVectorStore.cpp
    src/validation/ValidationService.cpp
    src/builder/CacheBuilderWorker.cpp
    src/builder/AdmissionController.cpp
    src/builder/IntelligentAdmissionPolicy.cpp
    src/builder/ResponseQualityFilter.cpp
    src/builder/Templatizer.cpp
    src/embedding/EmbeddingClient.cpp
    src/llm/OpenAIAdapter.cpp
    src/api/HttpServer.cpp
    src/quantization/TurboQuantizer.cpp
)

add_library(lettucecache_lib STATIC ${LIB_SOURCES})

target_include_directories(lettucecache_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${HIREDIS_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(lettucecache_lib PUBLIC
    ${HIREDIS_LIBRARIES}
    CURL::libcurl
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    httplib::httplib
    spdlog::spdlog
    pthread
)

if(faiss_FOUND)
    target_link_libraries(lettucecache_lib PUBLIC faiss)
else()
    target_include_directories(lettucecache_lib PUBLIC ${FAISS_INCLUDE_DIRS})
    target_link_libraries(lettucecache_lib PUBLIC ${FAISS_LIBRARIES})
endif()

# ── Executable ────────────────────────────────────────────────────────────
add_executable(lettucecache src/main.cpp)
target_link_libraries(lettucecache PRIVATE lettucecache_lib)

# ── Tests ─────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests/unit)
add_subdirectory(tests/integration)
