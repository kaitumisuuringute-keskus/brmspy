#!/usr/bin/env bash
set -euo pipefail

# run_main_tests.sh
# Run pytest with coverage collection only, then combine and report.

# Decide runner: uv if available, else python -m
if command -v uv >/dev/null 2>&1; then
  echo "[run_main_tests] Using 'uv run'"
  PYTEST_CMD=(uv run pytest)
  COVERAGE_CMD=(uv run coverage)
else
  echo "[run_main_tests] 'uv' not found, using 'python -m'"
  PYTEST_CMD=(python -m pytest)
  COVERAGE_CMD=(python -m coverage)
fi

# 1) Run tests, collect coverage but DO NOT print report from pytest
"${PYTEST_CMD[@]}" tests/ -v \
  --cov=brmspy \
  --cov-config=.coveragerc \
  --cov-report= \

# 2) Combine main-process .coverage + all .coverage.* shards
#    --append: include existing .coverage in the union
#    --keep:   keep .coverage.* for debugging/inspection
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" combine --append --keep

# 3) Final reports (only these are shown)
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" report -m   # term
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" xml         # coverage.xml
COVERAGE_FILE=.coverage "${COVERAGE_CMD[@]}" json        # coverage.json
