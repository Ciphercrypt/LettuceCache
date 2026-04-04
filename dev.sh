#!/usr/bin/env bash
# dev.sh - start LettuceCache locally (Redis + sidecar + binary)
# Usage: ./dev.sh [stop]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SIDECAR_DIR="$ROOT/python_sidecar"
VENV="$SIDECAR_DIR/.venv/bin/activate"
LOG_DIR="$ROOT/.dev-logs"
mkdir -p "$LOG_DIR"

PID_REDIS="$LOG_DIR/redis.pid"
PID_SIDECAR="$LOG_DIR/sidecar.pid"
PID_BINARY="$LOG_DIR/binary.pid"

kill_port() {
    # Kill all processes listening on a TCP port. SIGTERM first, SIGKILL fallback.
    local port="$1"
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null) || return 0
    [ -z "$pids" ] && return 0
    # Unblock httplib accept() so SIGTERM is processed immediately
    [ "$port" = "8080" ] && curl -sf "http://localhost:$port/health" >/dev/null 2>&1 || true
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    local i
    for i in 1 2 3; do
        pids=$(lsof -ti :"$port" 2>/dev/null) || break
        [ -z "$pids" ] && break
        sleep 1
    done
    pids=$(lsof -ti :"$port" 2>/dev/null) || return 0
    [ -n "$pids" ] && echo "$pids" | xargs kill -KILL 2>/dev/null || true
}

stop_all() {
    echo "Stopping..."
    kill_port 8080   # lettucecache binary
    kill_port 8001   # python sidecar
    kill_port 6379   # redis
    rm -f "$PID_BINARY" "$PID_SIDECAR" "$PID_REDIS"
    echo "All stopped."
    exit 0
}

[ "${1:-}" = "stop" ] && stop_all

# 1. Redis
echo "[1/3] Starting Redis..."
redis-server --daemonize yes \
    --pidfile "$PID_REDIS" \
    --logfile "$LOG_DIR/redis.log" \
    --appendonly no \
    --maxmemory 512mb \
    --maxmemory-policy allkeys-lru

# wait for Redis
for i in $(seq 1 10); do
    redis-cli ping &>/dev/null && break
    sleep 0.3
done
redis-cli ping &>/dev/null || { echo "Redis failed to start"; exit 1; }
echo "    Redis ready on :6379"

# 2. Python sidecar
echo "[2/3] Starting Python sidecar (first run downloads model ~90MB)..."
source "$VENV"
cd "$SIDECAR_DIR"
uvicorn main:app --port 8001 --log-level warning \
    >"$LOG_DIR/sidecar.log" 2>&1 &
echo $! > "$PID_SIDECAR"

# wait for sidecar
echo -n "    Waiting for sidecar"
for i in $(seq 1 30); do
    curl -sf http://localhost:8001/health &>/dev/null && break
    echo -n "."
    sleep 2
done
echo ""
curl -sf http://localhost:8001/health &>/dev/null || { echo "Sidecar failed - check $LOG_DIR/sidecar.log"; exit 1; }
echo "    Sidecar ready on :8001"

# 3. C++ binary
echo "[3/3] Starting LettuceCache..."
cd "$ROOT"
REDIS_HOST=localhost \
REDIS_PORT=6379 \
EMBED_URL=http://localhost:8001 \
EMBED_DIM=384 \
HTTP_PORT=8080 \
FAISS_INDEX_PATH="$LOG_DIR/faiss.index" \
OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
CACHE_QUALITY_THRESHOLD="${CACHE_QUALITY_THRESHOLD:-0.0}" \
./build/lettucecache >"$LOG_DIR/binary.log" 2>&1 &
echo $! > "$PID_BINARY"
sleep 1

curl -sf http://localhost:8080/health &>/dev/null || { echo "Binary failed - check $LOG_DIR/binary.log"; exit 1; }
echo "    LettuceCache ready on :8080"

echo ""
echo "All running. Logs in $LOG_DIR/"
echo "  Health : curl http://localhost:8080/health"
echo "  Stop   : ./dev.sh stop"
