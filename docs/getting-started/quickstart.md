# Quick Start

Get LettuceCache running locally in under 5 minutes using Docker Compose.

## Prerequisites

- Docker 24+ and Docker Compose v2
- An OpenAI API key (or leave blank to use the no-key fallback for testing)

## Steps

### 1. Clone the repo

```bash
git clone git@github.com:Ciphercrypt/LettuceCache.git
cd LettuceCache
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your API key:

```bash
OPENAI_API_KEY=sk-...
```

Everything else works with defaults for local development.

### 3. Start all services

```bash
docker compose up
```

This starts:

- **Redis 7** on `:6379`
- **Python sidecar** (embedding service) on `:8001`
- **C++ orchestrator** on `:8080`

Wait until you see:

```
orchestrator  | [INFO] HttpServer listening on port 8080
```

### 4. Verify it's healthy

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "redis": true,
  "embedding_sidecar": true,
  "faiss_entries": 0,
  "queue_depth": 0
}
```

### 5. Send your first query

```bash
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the capital of France?",
    "context": [],
    "domain": "geography"
  }' | jq .
```

The first call hits the LLM:

```json
{
  "answer": "The capital of France is Paris.",
  "cache_hit": false,
  "confidence": 0.0,
  "latency_ms": 712
}
```

Send the same query again:

```json
{
  "answer": "The capital of France is Paris.",
  "cache_hit": true,
  "confidence": 0.94,
  "latency_ms": 19
}
```

**37× faster on the second call.**

---

## What's Next

- [Configuration](configuration.md) — tune thresholds, models, and timeouts
- [How It Works](../how-it-works/overview.md) — understand context signatures and scoring
- [API Reference](../api/endpoints.md) — explore all endpoints
