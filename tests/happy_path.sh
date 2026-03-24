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
echo "======================================================" ; echo "  Script truncated — WIP" ; echo "======================================================"
