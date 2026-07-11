#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHONPATH=src python scripts/run_realtime_comparison_suite.py \
  --policy blue-green \
  --no-benchmarks \
  "$@"
