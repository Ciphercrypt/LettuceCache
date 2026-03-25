#!/usr/bin/env bash
# =============================================================================
# LettuceCache Live Demo
#
# Runs a real end-to-end demo with a live LLM (gpt-4o-mini by default) and
# Redis persistence. Walks through the full cache lifecycle:
#
#   1. Cold query  → LLM is called, response streamed back
#   2. Warm query  → AdmissionController admits, entry written to FAISS+Redis
#   3. L1 hit      → Redis exact match, 0ms
#   4. L2 hit      → FAISS+TurboQuant semantic match after L1 expires
#   5. Isolation   → Different user cannot hit another user's cached entry
#   6. Persistence → Restart the orchestrator, verify FAISS+Redis survive
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   ./tests/live_demo.sh
#
# Optional overrides:
#   LLM_MODEL=gpt-4o-mini  (default)
#   REDIS_PERSIST=1        (configures Redis RDB snapshots, default 1)
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
API="http://localhost:8080"
BOLD='\033[1m'; DIM='\033[2m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

FAISS_PATH="/tmp/lc_live_demo.faiss"

section() { echo -e "\n${BOLD}${CYAN}══ $1 ${NC}"; }
step()    { echo -e "${YELLOW}▶${NC} $1"; }
result()  { echo -e "${GREEN}✓${NC} $1"; }
warn()    { echo -e "${RED}✗${NC} $1"; }

cleanup() {
    echo -e "\n${DIM}Stopping stack...${NC}"
    pkill -f lettucecache   2>/dev/null || true
    pkill -f "uvicorn main" 2>/dev/null || true
    brew services stop redis 2>/dev/null | tail -1 || true
}
trap cleanup EXIT

# ── Validate prerequisites ─────────────────────────────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY is not set.${NC}"
    echo "  Run:  export OPENAI_API_KEY=sk-..."
    echo "  Then: ./tests/live_demo.sh"
    exit 1
fi

LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
export PATH="/opt/homebrew/opt/cmake/bin:/opt/homebrew/bin:$PATH"

# ── Helper: pretty-print a cache response ─────────────────────────────────
print_response() {
    local label="$1" json="$2"
    local hit conf lat answer entry
    hit=$(echo "$json"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['cache_hit'])")
    conf=$(echo "$json"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(round(d['confidence'],3))")
    lat=$(echo "$json"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['latency_ms'])")
    entry=$(echo "$json"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['cache_entry_id'] or '—')")
    answer=$(echo "$json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['answer'][:200])")

    if [ "$hit" = "True" ]; then
        local source="L1" icon="⚡"
        if ! echo "$entry" | grep -q "^lc:l1:"; then
            source="L2 (FAISS+TurboQuant)"; icon="🎯"
        fi
        echo -e "  ${GREEN}${icon} CACHE HIT [${source}]${NC}  confidence=${conf}  latency=${lat}ms"
    else
        echo -e "  ${YELLOW}→ LLM CALL${NC}  latency=${lat}ms"
    fi
    echo -e "  ${DIM}Answer: ${answer}${NC}"
}

query() {
    curl -s -X POST "$API/query" -H "Content-Type: application/json" -d "$1"
}

wait_for_sidecar() {
    for i in $(seq 1 25); do
        local r; r=$(curl -s --connect-timeout 1 "http://localhost:8001/health" 2>/dev/null || true)
        [ -n "$r" ] && return 0
        sleep 1
    done
    warn "Sidecar not reachable after 25s"
    exit 1
}

# ═════════════════════════════════════════════════════════════════════════════
echo -e "${BOLD}"
echo "  ██╗     ███████╗████████╗████████╗██╗   ██╗ ██████╗███████╗"
echo "  ██║     ██╔════╝╚══██╔══╝╚══██╔══╝██║   ██║██╔════╝██╔════╝"
echo "  ██║     █████╗     ██║      ██║   ██║   ██║██║     █████╗  "
echo "  ██║     ██╔══╝     ██║      ██║   ██║   ██║██║     ██╔══╝  "
echo "  ███████╗███████╗   ██║      ██║   ╚██████╔╝╚██████╗███████╗"
echo "  ╚══════╝╚══════╝   ╚═╝      ╚═╝    ╚═════╝  ╚═════╝╚══════╝"
echo -e "${NC}  Context-aware semantic cache for LLM calls"
echo -e "  LLM: ${CYAN}${LLM_MODEL}${NC}  |  TurboQuant: ${CYAN}enabled${NC}  |  Sidecar: ${CYAN}all-MiniLM-L6-v2${NC}\n"

# ═════════════════════════════════════════════════════════════════════════════
section "1. Starting Stack"
# ═════════════════════════════════════════════════════════════════════════════
step "Redis with RDB persistence..."
brew services start redis > /dev/null 2>&1 || true
sleep 1
# Enable RDB snapshot every 60s if ≥1 write
redis-cli config set save "60 1" > /dev/null
redis-cli config set dbfilename "lettucecache_demo.rdb" > /dev/null
redis-cli FLUSHALL > /dev/null
result "Redis ready  $(redis-cli INFO server | grep redis_version | tr -d '\r')"

step "Fresh FAISS index..."
rm -f "$FAISS_PATH" "${FAISS_PATH}.meta.json"

step "Python embedding sidecar (all-MiniLM-L6-v2)..."
cd "$REPO/python_sidecar"
source .venv/bin/activate
nohup uvicorn main:app --port 8001 --workers 1 > /tmp/lc_live_sidecar.log 2>&1 &
wait_for_sidecar
result "Sidecar ready  (dim=384)"

step "C++ orchestrator..."
REDIS_HOST=localhost REDIS_PORT=6379 EMBED_URL=http://localhost:8001 \
OPENAI_API_KEY="$OPENAI_API_KEY" LLM_MODEL="$LLM_MODEL" \
FAISS_INDEX_PATH="$FAISS_PATH" EMBED_DIM=384 \
HTTP_PORT=8080 ENABLE_TURBO_QUANT=1 \
nohup "$REPO/build/lettucecache" > /tmp/lc_live_orch.log 2>&1 &
sleep 2
HEALTH=$(curl -s "$API/health")
result "Orchestrator ready  redis=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['redis'])")  sidecar=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['embedding_sidecar'])")"

# ═════════════════════════════════════════════════════════════════════════════
section "2. Cold Queries — LLM Responds (admission threshold: 2 hits)"
# ═════════════════════════════════════════════════════════════════════════════
step "Query 1/2: 'What is the difference between L1 and L2 cache?'"
R=$(query '{"query":"What is the difference between L1 and L2 cache?","user_id":"alice","domain":"tech","correlation_id":"demo-1"}')
print_response "q1" "$R"

step "Query 2/2: same query (triggers admission → cache write)"
R=$(query '{"query":"What is the difference between L1 and L2 cache?","user_id":"alice","domain":"tech","correlation_id":"demo-2"}')
print_response "q2" "$R"

step "Waiting for async cache build..."
sleep 2
FAISS_COUNT=$(curl -s "$API/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['faiss_entries'])")
result "FAISS entries: ${FAISS_COUNT}  |  Redis L1 keys: $(redis-cli KEYS 'lc:l1:*' | wc -l | tr -d ' ')  |  Slot keys: $(redis-cli KEYS 'lc:slots:*' | wc -l | tr -d ' ')"

# ═════════════════════════════════════════════════════════════════════════════
section "3. L1 Cache Hit — Redis Exact Match (sub-millisecond)"
# ═════════════════════════════════════════════════════════════════════════════
step "Same query, same user, same domain..."
R=$(query '{"query":"What is the difference between L1 and L2 cache?","user_id":"alice","domain":"tech","correlation_id":"demo-3"}')
print_response "q3" "$R"

# Intentional paraphrase with same intent keywords
step "Paraphrase: 'difference L1 L2 cache' (same 3-keyword intent)"
R=$(query '{"query":"difference L1 L2 cache","user_id":"alice","domain":"tech","correlation_id":"demo-4"}')
print_response "q4" "$R"

# ═════════════════════════════════════════════════════════════════════════════
section "4. Second Topic — Python async programming"
# ═════════════════════════════════════════════════════════════════════════════
step "Query 1/2: 'How does async/await work in Python?'"
R=$(query '{"query":"How does async/await work in Python?","user_id":"alice","domain":"tech","correlation_id":"demo-5"}')
print_response "q5" "$R"
step "Query 2/2 (triggers admission)..."
R=$(query '{"query":"How does async/await work in Python?","user_id":"alice","domain":"tech","correlation_id":"demo-6"}')
print_response "q6" "$R"
sleep 2
result "FAISS entries: $(curl -s "$API/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['faiss_entries'])")"

# ═════════════════════════════════════════════════════════════════════════════
section "5. L2 Semantic Hit — FAISS + TurboQuant (after simulated L1 expiry)"
# ═════════════════════════════════════════════════════════════════════════════
step "Deleting L1 keys to simulate TTL expiry..."
for k in $(redis-cli KEYS "lc:l1:*"); do redis-cli DEL "$k" > /dev/null; done
result "L1 keys cleared. FAISS still has $(curl -s "$API/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['faiss_entries'])") vectors."

step "Re-query: 'What is the difference between L1 and L2 cache?' (L2 path)"
R=$(query '{"query":"What is the difference between L1 and L2 cache?","user_id":"alice","domain":"tech","correlation_id":"demo-7"}')
print_response "q7 (L2)" "$R"
echo -e "  ${DIM}entry_id: $(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin)['cache_entry_id'])")${NC}"

step "Re-query: 'How does async/await work in Python?' (L2 path)"
R=$(query '{"query":"How does async/await work in Python?","user_id":"alice","domain":"tech","correlation_id":"demo-8"}')
print_response "q8 (L2)" "$R"

# ═════════════════════════════════════════════════════════════════════════════
section "6. Context Isolation — Cache is Per-User"
# ═════════════════════════════════════════════════════════════════════════════
step "bob asks the same question as alice (should hit LLM, NOT alice's cache)..."
R=$(query '{"query":"What is the difference between L1 and L2 cache?","user_id":"bob","domain":"tech","correlation_id":"demo-9"}')
print_response "q9 (bob)" "$R"
echo -e "  ${DIM}(alice's answer is isolated to alice — bob gets a fresh LLM call)${NC}"

# ═════════════════════════════════════════════════════════════════════════════
section "7. Persistence Test — Restart Orchestrator, Cache Survives"
# ═════════════════════════════════════════════════════════════════════════════
step "Stopping orchestrator (FAISS + metadata will be written to disk)..."
pkill -f lettucecache 2>/dev/null || true
sleep 2

L1_BEFORE=$(redis-cli KEYS "lc:l1:*" | wc -l | tr -d ' ')
SLOT_BEFORE=$(redis-cli KEYS "lc:slots:*" | wc -l | tr -d ' ')
echo -e "  ${DIM}Redis L1 keys: ${L1_BEFORE}  |  Slot keys: ${SLOT_BEFORE}${NC}"
echo -e "  ${DIM}FAISS index written to: ${FAISS_PATH}  meta: ${FAISS_PATH}.meta.json${NC}"
ls -lh "$FAISS_PATH" "${FAISS_PATH}.meta.json" 2>/dev/null | awk '{print "  " $5, $9}'

step "Restarting orchestrator (loads FAISS + metadata from disk)..."
REDIS_HOST=localhost REDIS_PORT=6379 EMBED_URL=http://localhost:8001 \
OPENAI_API_KEY="$OPENAI_API_KEY" LLM_MODEL="$LLM_MODEL" \
FAISS_INDEX_PATH="$FAISS_PATH" EMBED_DIM=384 \
HTTP_PORT=8080 ENABLE_TURBO_QUANT=1 \
nohup "$REPO/build/lettucecache" > /tmp/lc_live_orch2.log 2>&1 &
sleep 3

FAISS_AFTER=$(curl -s "$API/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['faiss_entries'])")
result "Orchestrator restarted. FAISS entries loaded from disk: ${FAISS_AFTER}"

step "Query after restart (L2 must still work — metadata restored from disk)..."
# L1 backfill from the previous L2 hit is still in Redis
R=$(query '{"query":"What is the difference between L1 and L2 cache?","user_id":"alice","domain":"tech","correlation_id":"demo-10"}')
print_response "q10 (post-restart)" "$R"

# ═════════════════════════════════════════════════════════════════════════════
section "8. Summary"
# ═════════════════════════════════════════════════════════════════════════════
FINAL_HEALTH=$(curl -s "$API/health")
echo ""
echo -e "  ${BOLD}Stack state at end of demo:${NC}"
echo -e "  Redis keys      : $(redis-cli DBSIZE) total  ($(redis-cli KEYS 'lc:l1:*' | wc -l | tr -d ' ') L1, $(redis-cli KEYS 'lc:slots:*' | wc -l | tr -d ' ') slot)"
echo -e "  FAISS entries   : $(echo "$FINAL_HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['faiss_entries'])")"
echo -e "  LLM model       : ${LLM_MODEL}"
echo -e "  Embedding model : all-MiniLM-L6-v2 (384-dim, L2-normalised)"
echo -e "  TurboQuant      : 4-bit (3 MSE + 1 QJL)  →  244 bytes/vector vs 1536 bytes FP32"
echo ""
echo -e "  ${GREEN}${BOLD}Demo complete.${NC} Run 'redis-cli KEYS lc:*' to inspect all cache keys."
echo ""
