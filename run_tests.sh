#!/usr/bin/env bash
set -euo pipefail

# run_main_tests.sh
# Run pytest with coverage collection only, then combine and report.

# Decide runner: uv if available, else python -m
if command -v uv >/dev/null 2>&1; then
  echo "[run_main_tests] Using 'uv run'"
  PYTEST_CMD=(uv run pytest)
  COVERAGE_CMD=(uv run coverage)
  IMPORTLINTER_CMD=(uv run lint-imports)
else
  echo "[run_main_tests] 'uv' not found, using 'python -m'"
  PYTEST_CMD=(python -m pytest)
  COVERAGE_CMD=(python -m coverage)
  IMPORTLINTER_CMD=(lint-imports)
fi

TESTS="tests/"
RCFILE=".coveragerc"

# ---- best-effort raise nofile (soft + hard), never fail ----
(
  TARGET_NOFILE=500000

  # Read current hard limit (empty if unsupported)
  HARD="$(ulimit -Hn 2>/dev/null || true)"

  if [[ -n "${HARD}" ]]; then
    # Try to raise hard limit first (may be capped or denied)
    ulimit -Hn "${TARGET_NOFILE}" >/dev/null 2>&1 || true

    # Re-read hard limit (actual value after attempt)
    HARD="$(ulimit -Hn 2>/dev/null || true)"

    # Set soft limit to whatever hard ended up being
    if [[ -n "${HARD}" ]]; then
      ulimit -Sn "${HARD}" >/dev/null 2>&1 || true
    fi
  fi

  # Final state log (always best-effort)
  SOFT="$(ulimit -Sn 2>/dev/null || echo '?')"
  HARD="$(ulimit -Hn 2>/dev/null || echo '?')"
  echo "[nofile] soft=${SOFT} hard=${HARD}"
) || true
# -----------------------------------------------------------

# 1) Run tests, collect coverage but DO NOT print report from pytest
"${PYTEST_CMD[@]}" "$TESTS" -v \
  --cov=brmspy \
  --cov-config="$RCFILE" \
  --cov-report= \

# 2) Combine main-process .coverage + all .coverage.* shards
#    --append: include existing .coverage in the union
#    --keep:   keep .coverage.* for debugging/inspection
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" combine --append --keep --rcfile="$RCFILE"

# 3) Final reports (only these are shown)
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" report -m --rcfile="$RCFILE"   # term
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" xml --rcfile="$RCFILE"         # coverage.xml
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" json --rcfile="$RCFILE"        # coverage.json


"${IMPORTLINTER_CMD[@]}"