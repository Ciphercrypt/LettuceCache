#!/usr/bin/env bash
# =============================================================================
# LettuceCache Happy Path Integration Test
#
# Exercises the full HTTP API against the live stack:
#   Redis → Python sidecar → C++ orchestrator
#
# Usage:
#   ./tests/happy_path.sh               # builds + starts stack automatically
#   SKIP_BUILD=1 ./tests/happy_path.sh  # skip cmake build step
#
# Expected final output:  "All N tests passed."
# =============================================================================

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
API="http://localhost:8080"
PASS=0
FAIL=0
ERRORS=()

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  PASS${NC}  $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}  FAIL${NC}  $1"; FAIL=$((FAIL+1)); ERRORS+=("$1"); }
info() { echo -e "${YELLOW}  ----${NC}  $1"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
query() {
    # query <json_body> → full JSON response
    curl -s -X POST "$API/query" \
        -H "Content-Type: application/json" \
        -d "$1"
}

get_field() {
    # get_field <json> <field>
    echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['$2'])"
}

assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [ "$actual" = "$expected" ]; then
        ok "$label (got: $actual)"
    else
        fail "$label (expected: $expected, got: $actual)"
    fi
}

assert_lt() {
    local label="$1" threshold="$2" actual="$3"
    if python3 -c "exit(0 if float('$actual') < $threshold else 1)" 2>/dev/null; then
        ok "$label (got: $actual < $threshold)"
    else
        fail "$label (expected < $threshold, got: $actual)"
    fi
}

assert_gt() {
    local label="$1" threshold="$2" actual="$3"
    if python3 -c "exit(0 if float('$actual') > $threshold else 1)" 2>/dev/null; then
        ok "$label (got: $actual > $threshold)"
    else
        fail "$label (expected > $threshold, got: $actual)"
    fi
}

assert_contains() {
    local label="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        ok "$label"
    else
        fail "$label (expected '$needle' in '$haystack')"
    fi
}

assert_not_contains() {
    local label="$1" needle="$2" haystack="$3"
    if ! echo "$haystack" | grep -qF "$needle"; then
        ok "$label"
    else
        fail "$label (unexpected '$needle' found in '$haystack')"
    fi
}

wait_for() {
    local url="$1" label="$2" retries="${3:-20}"
    for i in $(seq 1 "$retries"); do
        local r
        r=$(curl -s --connect-timeout 1 "$url" 2>/dev/null || true)
        if [ -n "$r" ]; then echo "$r"; return 0; fi
        sleep 1
    done
    fail "$label: not reachable after ${retries}s"
    return 1
}

cleanup() {
    info "Stopping stack..."
    pkill -f lettucecache   2>/dev/null || true
    pkill -f "uvicorn main" 2>/dev/null || true
    brew services stop redis 2>/dev/null | tail -1 || true
    rm -f "${FAISS_PATH:-}" "${FAISS_PATH:-}.meta.json" 2>/dev/null || true
}
trap cleanup EXIT

# ═════════════════════════════════════════════════════════════════════════════
# 0. Build
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════"
echo "  LettuceCache Happy Path Test"
echo "══════════════════════════════════════════════════════"
echo ""

export PATH="/opt/homebrew/opt/cmake/bin:/opt/homebrew/bin:$PATH"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    info "Building..."
    cmake -B "$REPO/build" -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_PREFIX_PATH="/opt/homebrew" -DCMAKE_CXX_COMPILER=clang++ \
          -Wno-dev > /tmp/lc_cmake.log 2>&1
    cmake --build "$REPO/build" --target lettucecache \
          -j"$(sysctl -n hw.ncpu)" > /tmp/lc_build.log 2>&1
    ok "Build succeeded"
fi

# ═════════════════════════════════════════════════════════════════════════════
# 1. Start services
# ═════════════════════════════════════════════════════════════════════════════
info "Starting Redis..."
brew services start redis > /dev/null 2>&1
sleep 1
redis-cli ping | grep -q PONG && ok "Redis started" || fail "Redis failed to start"
redis-cli FLUSHALL > /dev/null

# Fresh FAISS index every run: if we load a stale index, its slot keys are
# gone from Redis (FLUSHALL) and L2 hits would return raw templates.
FAISS_PATH="/tmp/lc_happy_path_$$.faiss"
rm -f "$FAISS_PATH" "${FAISS_PATH}.meta.json"

info "Starting Python sidecar..."
cd "$REPO/python_sidecar"
source .venv/bin/activate
nohup uvicorn main:app --port 8001 --workers 1 > /tmp/lc_sidecar.log 2>&1 &

info "Waiting for sidecar (model load ~10s)..."
SIDECAR_HEALTH=$(wait_for "http://localhost:8001/health" "Python sidecar" 30)
SIDECAR_DIM=$(get_field "$SIDECAR_HEALTH" "dimension")
assert_eq "Sidecar healthy, dim=384" "384" "$SIDECAR_DIM"

info "Starting C++ orchestrator (TurboQuant enabled)..."
REDIS_HOST=localhost REDIS_PORT=6379 \
EMBED_URL=http://localhost:8001 OPENAI_API_KEY="" \
FAISS_INDEX_PATH="$FAISS_PATH" EMBED_DIM=384 \
HTTP_PORT=8080 ENABLE_TURBO_QUANT=1 \
nohup "$REPO/build/lettucecache" > /tmp/lc_orch.log 2>&1 &
sleep 2

# ═════════════════════════════════════════════════════════════════════════════
# 2. Health check
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 2. Health check ──────────────────────────────────"
HEALTH=$(curl -s "$API/health")
assert_eq "Redis healthy"            "True"  "$(get_field "$HEALTH" "redis")"
assert_eq "Sidecar healthy"          "True"  "$(get_field "$HEALTH" "embedding_sidecar")"
assert_eq "Status ok"                "ok"    "$(get_field "$HEALTH" "status")"
assert_eq "FAISS empty at start"     "0"     "$(get_field "$HEALTH" "faiss_entries")"

# ═════════════════════════════════════════════════════════════════════════════
# 3. Cache MISS — first-ever query
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 3. Cache MISS (first query) ──────────────────────"
Q='{"query":"What is machine learning?","user_id":"alice","domain":"tech","correlation_id":"hp-1"}'
R=$(query "$Q")
assert_eq "cache_hit=false on first query"    "False" "$(get_field "$R" "cache_hit")"
assert_eq "confidence=0.0 on miss"            "0.0"   "$(get_field "$R" "confidence")"
assert_not_contains "No SLOT placeholders in miss response" "{{SLOT_" "$(get_field "$R" "answer")"

# ═════════════════════════════════════════════════════════════════════════════
# 4. Admission gate — second hit triggers cache write
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 4. Admission gate (second hit → async cache write) ──"
R2=$(query "$Q")
assert_eq "cache_hit=false before admission write completes" "False" "$(get_field "$R2" "cache_hit")"
sleep 1  # let CacheBuilderWorker flush
HEALTH2=$(curl -s "$API/health")
assert_eq "FAISS has 1 entry after admission" "1" "$(get_field "$HEALTH2" "faiss_entries")"
assert_eq "Build queue drained"               "0" "$(get_field "$HEALTH2" "queue_depth")"

# ═════════════════════════════════════════════════════════════════════════════
# 5. L1 HIT — identical query, exact Redis match
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 5. L1 HIT (exact Redis match) ────────────────────"
R3=$(query "$Q")
assert_eq "cache_hit=true on L1"         "True" "$(get_field "$R3" "cache_hit")"
assert_eq "confidence=1.0 on L1 hit"    "1.0"  "$(get_field "$R3" "confidence")"
assert_lt "L1 latency < 5ms"            "5"    "$(get_field "$R3" "latency_ms")"
assert_not_contains "L1 answer has no placeholders" "{{SLOT_" "$(get_field "$R3" "answer")"

L1_ANSWER=$(get_field "$R3" "answer")

# ═════════════════════════════════════════════════════════════════════════════
# 6. Same intent, different stopwords → same L1 key
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 6. Same intent, different phrasing → same L1 key ──"
# "What is machine learning?" and "machine learning" → intent: "machine_learning"
Q_PARAPHRASE='{"query":"machine learning","user_id":"alice","domain":"tech","correlation_id":"hp-6"}'
R6=$(query "$Q_PARAPHRASE")
assert_eq "cache_hit=true for same-intent paraphrase" "True" "$(get_field "$R6" "cache_hit")"
assert_eq "confidence=1.0 (L1)"                       "1.0"  "$(get_field "$R6" "confidence")"

# ═════════════════════════════════════════════════════════════════════════════
# 7. L2 HIT — simulate L1 TTL expiry, FAISS + TurboQuant serves from vector
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 7. L2 HIT (after L1 expiry, TurboQuant unbiased IP) ──"
L1_SIG_KEY=$(redis-cli KEYS "lc:l1:*" | head -1)
redis-cli DEL "$L1_SIG_KEY" > /dev/null
R7=$(query "$Q")
assert_eq  "cache_hit=true on L2"                   "True" "$(get_field "$R7" "cache_hit")"
assert_gt  "TurboQuant confidence > 0.8"            "0.8"  "$(get_field "$R7" "confidence")"
assert_not_contains "L2 answer rendered (no SLOT_)"  "{{SLOT_" "$(get_field "$R7" "answer")"
assert_eq  "L2 answer equals original LLM response" "$L1_ANSWER" "$(get_field "$R7" "answer")"

L2_ENTRY_ID=$(get_field "$R7" "cache_entry_id")
# L2 hit returns FAISS entry ID (not lc:l1: prefix)
if echo "$L2_ENTRY_ID" | grep -qv "^lc:l1:"; then
    ok "L2 cache_entry_id is FAISS ID (not L1 key)"
else
    fail "L2 cache_entry_id should be FAISS ID, got: $L2_ENTRY_ID"
fi

# ═════════════════════════════════════════════════════════════════════════════
# 8. L1 backfill after L2 hit
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 8. L1 backfill (next identical query after L2 hit) ──"
R8=$(query "$Q")
assert_eq "cache_hit=true after backfill"  "True" "$(get_field "$R8" "cache_hit")"
assert_eq "confidence=1.0 (L1 backfill)"  "1.0"  "$(get_field "$R8" "confidence")"
assert_lt "L1 backfill latency < 5ms"     "5"    "$(get_field "$R8" "latency_ms")"

# ═════════════════════════════════════════════════════════════════════════════
# 9. Context isolation — different user must NOT hit alice's cache
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 9. Context isolation (different user → miss) ──────"
Q_BOB='{"query":"What is machine learning?","user_id":"bob","domain":"tech","correlation_id":"hp-9"}'
R9=$(query "$Q_BOB")
assert_eq "cache_hit=false for different user" "False" "$(get_field "$R9" "cache_hit")"

# ═════════════════════════════════════════════════════════════════════════════
# 10. Domain isolation — same user, different domain → miss
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 10. Domain isolation (different domain → miss) ────"
Q_DOMAIN='{"query":"What is machine learning?","user_id":"alice","domain":"biology","correlation_id":"hp-10"}'
R10=$(query "$Q_DOMAIN")
assert_eq "cache_hit=false for different domain" "False" "$(get_field "$R10" "cache_hit")"

# ═════════════════════════════════════════════════════════════════════════════
# 11. Context canonicalization — reversed context order → same L1 key
# ═════════════════════════════════════════════════════════════════════════════
echo ""
info "── 11. Context canonicalization (reordered turns → same key) ──"
QC_A='{"query":"What is machine learning?","user_id":"carol","domain":"tech","context":["user: hi","bot: hello"],"correlation_id":"hp-11a"}'
QC_B='{"query":"What is machine learning?","user_id":"carol","domain":"tech","context":["bot: hello","user: hi"],"correlation_id":"hp-11b"}'
# Prime carol's cache
query "$QC_A" > /dev/null
query "$QC_A" > /dev/null
sleep 1
# Query with reversed context — should hit same key
R11=$(query "$QC_B")
assert_eq "cache_hit=true for reordered context" "True" "$(get_field "$R11" "cache_hit")"

# ═════════════════════════════════════════════════════════════════════════════
# 12. Templatizer slot rendering — L2 serves filled response, not raw template
# ═════════════════════════════════════════════════════════════════════════════
