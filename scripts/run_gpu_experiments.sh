#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

export PYTHONPATH=src

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY

echo "Generating controlled ablation configs..."
"${PYTHON_BIN}" scripts/run_controlled_ablations.py --generate-only "$@"

echo
echo "Generated configs under configs/ablations."
echo "To run the full sweep:"
echo "  ${PYTHON_BIN} scripts/run_controlled_ablations.py $*"
echo
echo "To run a shorter smoke sweep first:"
echo "  ${PYTHON_BIN} scripts/run_controlled_ablations.py --epochs 3 --only baseline loss_false_safe augment_off resolution_720x448 class_weights_neutral"
echo
echo "To benchmark GPU forward inference after a checkpoint exists:"
echo "  ${PYTHON_BIN} scripts/gpu_inference_benchmark.py --config configs/blue_green_second_run.yaml --batch-size 8 --limit 128 --repeats 20"
