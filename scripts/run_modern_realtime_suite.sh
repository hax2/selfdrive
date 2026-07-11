#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

CONFIGS=(
  configs/segformer_b0_cat_blue.yaml
  configs/ddrnet23_slim_cat_blue.yaml
  configs/bisenetv2_cat_blue.yaml
)

if [[ ! -f data_processed_blue/summary.json || ! -f configs/discovered_mapping_blue.yaml ]]; then
  echo "Blue-only processed data is missing. Run: make prepare-rod-blue" >&2
  exit 1
fi

experiment_name() {
  PYTHONPATH=src python - "$1" <<'PY'
import sys
from pathlib import Path
from ttfm.utils import load_yaml

print(load_yaml(Path(sys.argv[1]))["experiment_name"])
PY
}

for config in "${CONFIGS[@]}"; do
  experiment="$(experiment_name "${config}")"
  metrics="outputs/${experiment}/test_metrics.json"

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${metrics}" ]]; then
    echo "Skipping completed experiment: ${experiment}"
  else
    echo "Running ${experiment}"
    bash scripts/run_train_eval_review.sh "${config}"
  fi

  if [[ "${RUN_BENCHMARKS}" == "1" ]]; then
    echo "Benchmarking ${experiment}"
    PYTHONPATH=src python scripts/gpu_inference_benchmark.py \
      --config "${config}" \
      --input-dir data_processed_blue/test/images \
      --device cuda \
      --batch-size 1 \
      --warmup 5 \
      --repeats 10 \
      --limit 128 \
      --output "reports/benchmark_${experiment}.json"
  fi
done

echo "Modern real-time suite completed successfully."
