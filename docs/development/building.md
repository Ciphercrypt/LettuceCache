# Building from Source

## Requirements

| Tool | Minimum version |
|---|---|
| CMake | 3.20 |
| C++ compiler | GCC 12+ or Clang 14+ |
| libfaiss | 1.7+ |
| libhiredis | 1.0+ |
| libcurl | 7.68+ |
| OpenSSL | 1.1+ |

The following are fetched automatically by CMake `FetchContent` if not found locally:

- `nlohmann/json` 3.11.3
- `cpp-httplib` 0.15.3
- `spdlog` 1.13.0

## Install System Dependencies

=== "macOS"

    ```bash
    brew install cmake faiss hiredis openssl curl pkg-config
    ```

=== "Ubuntu 22.04+"

    ```bash
    sudo apt-get update && sudo apt-get install -y \
        cmake build-essential pkg-config \
        libfaiss-dev libhiredis-dev \
        libcurl4-openssl-dev libssl-dev
    ```

## Configure and Build

```bash
# Release build (optimised)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Debug build
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j$(nproc)
```

## CMake Targets

| Target | Description |
|---|---|
| `lettucecache` | Main executable |
| `lettucecache_lib` | Static library — shared by exe and tests |
| `test_unit` | Unit test binary |
| `test_integration` | Integration test binary |

## Build the Python Sidecar

```bash
cd python_sidecar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Compile Commands (IDE integration)

CMake generates a `compile_commands.json` (enabled via `CMAKE_EXPORT_COMPILE_COMMANDS=ON`). Point clangd or your IDE at `build/compile_commands.json` for accurate IntelliSense.
