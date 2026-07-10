#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/third_run.yaml}"
POLL_SECONDS="${POLL_SECONDS:-15}"
TOP_K="${TOP_K:-10}"
TRAIN_PATTERN="ttfm.cli --config ${CONFIG_PATH} train"

echo "Watching for training process: ${TRAIN_PATTERN}"
echo "Polling every ${POLL_SECONDS}s"

while pgrep -af "${TRAIN_PATTERN}" >/dev/null; do
  date '+[%Y-%m-%d %H:%M:%S] training still running'
  sleep "${POLL_SECONDS}"
done

echo "Training process no longer running."
echo "Running eval..."
PYTHONPATH=src python -m ttfm.cli --config "${CONFIG_PATH}" eval

echo "Running review..."
PYTHONPATH=src python -m ttfm.cli --config "${CONFIG_PATH}" review --split test --top-k "${TOP_K}"

echo "Post-processing complete."
