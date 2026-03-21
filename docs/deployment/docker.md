# Docker Compose

The quickest way to run LettuceCache in a fully containerised setup.

## Services

```yaml
services:
  redis           # Redis 7 — L1 store + async stream
  python_sidecar  # FastAPI embedding service on :8001
  orchestrator    # C++ HTTP server on :8080
```

## Start

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env

docker compose up
```

Add `--build` to rebuild images after source changes:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up -d
docker compose logs -f orchestrator
```

## Stop

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop and delete volumes (wipes Redis + FAISS index)
```

## Volumes

| Volume | Mounted at | Purpose |
|---|---|---|
| `redis_data` | Redis container | Persistent AOF journal |
| `faiss_data` | `/data` in orchestrator | FAISS index file (`faiss.index`) |

The FAISS index persists across container restarts. To reset the index:

```bash
docker compose down -v
docker compose up
```

## Health Checks

All three services have Docker health checks. `python_sidecar` and `orchestrator` wait for `redis` to be healthy before starting.

```bash
docker compose ps
```

```
NAME                STATUS
lettucecache-redis-1            Up (healthy)
lettucecache-python_sidecar-1   Up (healthy)
lettucecache-orchestrator-1     Up (healthy)
```

## Environment Override

Override any variable without editing `.env`:

```bash
OPENAI_API_KEY=sk-... HTTP_PORT=9090 docker compose up
```

## Rebuild the C++ Image

The `Dockerfile` uses a multi-stage build:

1. **Builder stage** — installs all build deps, compiles with CMake
2. **Runtime stage** — copies only the binary + required libs into a slim image

```bash
docker compose build orchestrator
```

Build arguments can be passed for custom CMake flags:

```bash
docker build --build-arg CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Debug" -t lettucecache:debug .
```
