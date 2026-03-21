# Installation

## Option A — Docker Compose (recommended)

No local C++ toolchain needed. Requires Docker 24+ and Docker Compose v2.

```bash
git clone git@github.com:Ciphercrypt/LettuceCache.git
cd LettuceCache
cp .env.example .env          # set OPENAI_API_KEY
docker compose up --build
```

---

## Option B — Build from Source

### System Dependencies

=== "macOS"

    ```bash
    brew install cmake faiss hiredis openssl curl pkg-config
    ```

=== "Ubuntu / Debian"

    ```bash
    sudo apt-get update
    sudo apt-get install -y \
        cmake build-essential pkg-config \
        libfaiss-dev libhiredis-dev \
        libcurl4-openssl-dev libssl-dev
    ```

=== "Arch Linux"

    ```bash
    sudo pacman -S cmake faiss hiredis curl openssl pkgconf
    ```

### Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The binary is at `build/lettucecache`.

### Run Dependencies

Start Redis and the Python sidecar (these are still easiest via Docker):

```bash
docker compose up redis python_sidecar
```

### Run the Orchestrator

```bash
export OPENAI_API_KEY=sk-...
export REDIS_HOST=localhost
export EMBED_URL=http://localhost:8001
./build/lettucecache
```

---

## Python Sidecar (standalone)

If you want to run the sidecar outside of Docker:

```bash
cd python_sidecar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 2
```

The first startup downloads the `all-MiniLM-L6-v2` model (~90 MB). Subsequent starts use the local cache.

---

## Verify the Build

```bash
# Unit tests — no external dependencies needed
cd build && ctest -R unit --output-on-failure
```

Expected output:

```
Test project .../build
    Start 1: test_context_signature
1/4 Test #1: test_context_signature ....... Passed  0.01s
    Start 2: test_validation
2/4 Test #2: test_validation .............. Passed  0.01s
    Start 3: test_admission_controller
3/4 Test #3: test_admission_controller .... Passed  0.01s
    Start 4: test_templatizer
4/4 Test #4: test_templatizer ............. Passed  0.01s

100% tests passed, 0 tests failed out of 4
```
