#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/fourth_run.yaml}"
TOP_K="${TOP_K:-10}"

echo "Running train with ${CONFIG_PATH}"
PYTHONPATH=src python -m ttfm.cli --config "${CONFIG_PATH}" train

echo "Running eval with ${CONFIG_PATH}"
PYTHONPATH=src python -m ttfm.cli --config "${CONFIG_PATH}" eval

echo "Running review with ${CONFIG_PATH}"
PYTHONPATH=src python -m ttfm.cli --config "${CONFIG_PATH}" review --split test --top-k "${TOP_K}"

echo "Done."
