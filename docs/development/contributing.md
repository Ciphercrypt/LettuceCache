# Contributing

## Project Structure

```
src/
├── orchestrator/   Hot path coordination — touch carefully
├── cache/          Redis + FAISS adapters
├── validation/     Scoring logic
├── builder/        Async write path
├── embedding/      HTTP client to Python sidecar
├── llm/            LLM adapter interface + implementations
└── api/            REST server — wires everything together
```

## Adding a New LLM Adapter

1. Subclass `llm::LLMAdapter` in `src/llm/`:

```cpp
class AnthropicAdapter : public LLMAdapter {
public:
    AnthropicAdapter(const std::string& api_key);
    std::string complete(const std::string& query,
                         const std::vector<std::string>& context) override;
};
```

2. Add the `.cpp` to `CMakeLists.txt` under `LIB_SOURCES`
3. Wire it in `HttpServer.cpp`

## Swapping the Embedding Model

Change `MODEL_NAME` in `.env`:

```bash
MODEL_NAME=all-mpnet-base-v2   # 768-dim; slower, more accurate
MODEL_NAME=all-MiniLM-L12-v2   # 384-dim; balanced
```

**Important:** Update `EMBED_DIM` to match the model's output dimension. FAISS and the orchestrator must agree on the dimension at startup.

## Code Style

- C++17; constructor injection only (no `@Autowired` equivalent — no global singletons)
- Namespaces follow directory structure: `lettucecache::cache`, `lettucecache::orchestrator`, etc.
- Structured logging with `spdlog` — always include `correlation_id` in log messages
- No raw owning pointers — use `std::unique_ptr` / `std::shared_ptr`

## Commit Style

Follow conventional commits:

```
feat(validation): add intent matching to composite score
fix(cache): handle Redis reconnect on ECONNRESET
docs: update API reference for POST /feedback
test: add templatizer edge cases for IPv6 addresses
```

## Documentation

Update `docs/` alongside any code change. The docs site is auto-deployed on every push to `main` via GitHub Actions.
