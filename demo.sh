#!/usr/bin/env bash
# demo.sh — end-to-end smoke test for LettuceCache
# Usage: ./demo.sh [host:port]   default: localhost:8080

set -euo pipefail

BASE="${1:-http://localhost:8080}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/.dev-logs"
FAILURES=0

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

pass()    { echo -e "  ${GREEN}PASS${RESET} $1"; }
fail()    { echo -e "  ${RED}FAIL${RESET} $1"; FAILURES=$((FAILURES+1)); }
info()    { echo -e "${DIM}       $1${RESET}"; }
section() { echo -e "\n${BOLD}${CYAN}$1${RESET}"; }

get_field() { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin)['$2'])"; }
assert_hit()  { [[ "$(get_field "$1" cache_hit)" == "True"  ]] && pass "$2" || fail "$2"; }
assert_miss() { [[ "$(get_field "$1" cache_hit)" == "False" ]] && pass "$2" || fail "$2"; }
latency()     { get_field "$1" latency_ms; }
confidence()  { echo "$1" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['confidence']:.3f}\")"; }
faiss_count() { curl -sf "$BASE/stats" | python3 -c "import sys,json; print(json.load(sys.stdin)['faiss_entries'])"; }

query() {
    local qtext="$1" extra="${2:-}"
    curl -sf -X POST "$BASE/query" -H 'Content-Type: application/json' -d \
        "{\"query\":\"$qtext\",\"domain\":\"tech\",\"temperature\":0.2,\"system_prompt\":\"You are a helpful assistant.\"${extra:+,$extra}}"
}

# ── 0. Health ─────────────────────────────────────────────────────────────────
section "0. Health check"
health=$(curl -sf "$BASE/health")
[[ "$(get_field "$health" redis)"             == "True" ]] && pass "Redis reachable"   || fail "Redis unreachable"
[[ "$(get_field "$health" embedding_sidecar)" == "True" ]] && pass "Sidecar reachable" || fail "Sidecar unreachable"

# ── 1. Clean slate ────────────────────────────────────────────────────────────
section "1. Clean slate — restart binary with fresh FAISS, flush Redis"

# Restart only the binary so FAISS in-memory state is cleared.
# Sidecar + Redis keep running; startup is instant.
if [[ -f "$LOG_DIR/binary.pid" ]]; then
    kill "$(cat "$LOG_DIR/binary.pid")" 2>/dev/null || true
    rm "$LOG_DIR/binary.pid"
fi
rm -f "$LOG_DIR/faiss.index" "$LOG_DIR/faiss.index.meta.json"
redis-cli FLUSHALL > /dev/null

REDIS_HOST=localhost REDIS_PORT=6379 EMBED_URL=http://localhost:8001 EMBED_DIM=384 \
HTTP_PORT=8080 FAISS_INDEX_PATH="$LOG_DIR/faiss.index" OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
CACHE_QUALITY_THRESHOLD=0.0 \
"$ROOT/build/lettucecache" > "$LOG_DIR/binary.log" 2>&1 &
echo $! > "$LOG_DIR/binary.pid"
sleep 1

curl -sf "$BASE/health" > /dev/null
pass "Binary restarted, FAISS empty, Redis flushed"
info "faiss_entries=$(faiss_count)"

# ── 2. Admission gate ─────────────────────────────────────────────────────────
section "2. Admission gate (min 2 misses before caching)"
r1=$(query "What is machine learning?")
assert_miss "$r1" "Request 1: miss  (latency $(latency "$r1") ms)"

r2=$(query "What is machine learning?")
assert_miss "$r2" "Request 2: miss  (latency $(latency "$r2") ms)"
info "Background worker: embed -> quality filter -> CVF -> FAISS add -> Redis MULTI/EXEC"
sleep 1
info "faiss_entries=$(faiss_count)"

# ── 3. L1 exact hit ──────────────────────────────────────────────────────────
section "3. L1 exact hit"
r3=$(query "What is machine learning?")
assert_hit "$r3" "Exact query: L1 hit  confidence=$(confidence "$r3")  latency=$(latency "$r3") ms"
info "Same SHA-256 key -> Redis GET, no embedding call, no FAISS lookup"

# ── 4. L2 semantic hit ────────────────────────────────────────────────────────
section "4. L2 semantic hit (paraphrased queries)"
r4=$(query "How does machine learning work?")
assert_hit "$r4" "Paraphrase 1: L2 hit  confidence=$(confidence "$r4")  latency=$(latency "$r4") ms"
ENTRY_ID="$(get_field "$r4" cache_entry_id)"

r5=$(query "Can you explain machine learning?")
assert_hit "$r5" "Paraphrase 2: L2 hit  confidence=$(confidence "$r5")  latency=$(latency "$r5") ms"
info "Different intent tokens, same context fingerprint -> cosine >= 0.85 -> FAISS hit"

# ── 5. Domain isolation ───────────────────────────────────────────────────────
section "5. Domain isolation"
r6=$(query "What is machine learning?" '"domain":"finance"')
assert_miss "$r6" "domain=finance, same query: miss  (fingerprint differs)"

# ── 6. System-prompt isolation ────────────────────────────────────────────────
section "6. System-prompt isolation"
r7=$(curl -sf -X POST "$BASE/query" -H 'Content-Type: application/json' -d \
    '{"query":"What is machine learning?","domain":"tech","temperature":0.2,"system_prompt":"You are a red-teamer."}')
assert_miss "$r7" "Different system_prompt: miss  (system_prompt hash differs)"

# ── 7. High-temperature bypass ────────────────────────────────────────────────
section "7. High-temperature bypass"
r8=$(curl -sf -X POST "$BASE/query" -H 'Content-Type: application/json' -d \
    '{"query":"What is machine learning?","domain":"tech","temperature":0.9,"system_prompt":"You are a helpful assistant."}')
assert_miss "$r8" "temperature=0.9: cache bypassed entirely  (latency $(latency "$r8") ms)"
info "temperature >= 0.7 skips cache read and write"

# ── 8. DELETE single entry ────────────────────────────────────────────────────
section "8. DELETE single entry"
before=$(faiss_count)
if [[ -n "$ENTRY_ID" && "$ENTRY_ID" != "None" ]]; then
    del=$(curl -sf -X DELETE "$BASE/cache/$ENTRY_ID")
    [[ "$(get_field "$del" deleted)" == "True" ]] \
        && pass "DELETE /cache/$ENTRY_ID: deleted=true" \
        || fail "DELETE returned deleted=false"
    sleep 0.2
    after=$(faiss_count)
    [[ "$after" -lt "$before" ]] \
        && pass "FAISS entries dropped from $before to $after after DELETE" \
        || fail "FAISS count did not decrease (before=$before after=$after)"
    info "Tombstone prevents the evicted entry from being served at L2 even before FAISS GC"
else
    info "No L2 entry_id captured — skipping DELETE test"
fi

# ── 9. Domain bulk invalidation ───────────────────────────────────────────────
section "9. Domain bulk invalidation"
for _ in 1 2; do
    curl -sf -X POST "$BASE/query" -H 'Content-Type: application/json' -d \
        '{"query":"What is compound interest?","domain":"finance","system_prompt":"You are a finance advisor.","temperature":0.1}' > /dev/null
done
sleep 1
pre=$(curl -sf -X POST "$BASE/query" -H 'Content-Type: application/json' -d \
    '{"query":"What is compound interest?","domain":"finance","system_prompt":"You are a finance advisor.","temperature":0.1}')
assert_hit "$pre" "Finance entry warm  confidence=$(confidence "$pre")"

before_bulk=$(faiss_count)
bulk=$(curl -sf -X DELETE "$BASE/cache/domain/finance")
info "DELETE /cache/domain/finance: removed=$(get_field "$bulk" removed) entries"
sleep 0.2
after_bulk=$(faiss_count)
[[ "$after_bulk" -lt "$before_bulk" ]] \
    && pass "FAISS entries dropped from $before_bulk to $after_bulk" \
    || fail "FAISS count did not decrease after domain DELETE"

post=$(curl -sf -X POST "$BASE/query" -H 'Content-Type: application/json' -d \
    '{"query":"What is compound interest?","domain":"finance","system_prompt":"You are a finance advisor.","temperature":0.1}')
assert_miss "$post" "After domain DELETE: finance miss"

# ── 10. Final stats ───────────────────────────────────────────────────────────
section "10. Final stats"
stats=$(curl -sf "$BASE/stats")
info "faiss_entries=$(get_field "$stats" faiss_entries)  queue_depth=$(get_field "$stats" queue_depth)"

echo ""
if [[ $FAILURES -eq 0 ]]; then
    echo -e "${BOLD}${GREEN}All assertions passed.${RESET}"
else
    echo -e "${BOLD}${RED}$FAILURES assertion(s) failed.${RESET}"
    exit 1
fi
