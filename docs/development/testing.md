# Running Tests

## Unit Tests

Pure in-memory — no Redis, no FAISS index file, no network. Run anywhere.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test_unit
cd build && ctest -R unit --output-on-failure
```

### What's Tested

| Test file | What it covers |
|---|---|
| `test_context_signature.cpp` | SHA-256 output stability, user ID anonymisation, different inputs produce different hashes |
| `test_validation.cpp` | Score formula correctness, threshold boundary conditions, cosine similarity edge cases |
| `test_admission_controller.cpp` | Frequency gate, sliding window expiry, size rejection |
| `test_templatizer.cpp` | UUID replacement, numeric ID extraction, email masking, date masking |

## Integration Tests

Require a live Redis on `localhost:6379`. Start it with:

```bash
docker compose up redis
```

Enable and run:

```bash
INTEGRATION_TESTS=1 cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target test_integration
cd build && ctest -R integration --output-on-failure
```

### What's Tested

| Test | What it covers |
|---|---|
| Redis set/get/del | `RedisCacheAdapter` round-trip |
| FAISS add/search/remove | Index write + nearest-neighbour retrieval + eviction |
| CacheBuilderWorker pipeline | End-to-end: enqueue → admission → templatize → FAISS write |

## Test Output

Passing run:

```
Test project /path/to/build
      Start 1: test_context_signature
  1/4 Test  #1: test_context_signature ....... Passed  0.01s
      Start 2: test_validation
  2/4 Test  #2: test_validation .............. Passed  0.01s
      Start 3: test_admission_controller
  3/4 Test  #3: test_admission_controller .... Passed  0.01s
      Start 4: test_templatizer
  4/4 Test  #4: test_templatizer ............. Passed  0.01s

100% tests passed, 0 tests failed out of 4
```

## Tips

- Run with `--verbose` for per-assertion output: `ctest --output-on-failure --verbose`
- Run a single test by name: `ctest -R test_validation`
- Tests use Google Test (fetched automatically by CMake if not found)
