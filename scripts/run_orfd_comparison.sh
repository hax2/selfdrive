#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

venv_dir="${ORFD_VENV:-.venv-orfd}"
python_bin="${PYTHON_BIN:-python3}"
seeds="${ORFD_SEEDS:-1337 2027 4242}"
raw_root="${ORFD_RAW_ROOT:-datasets/ORFD}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$python_bin" -m venv "$venv_dir"
fi
# shellcheck disable=SC1091
source "$venv_dir/bin/activate"

python -m pip --isolated install --no-user --upgrade pip
python -m pip --isolated install --no-user -r requirements.txt
python -m pip --isolated install --no-user libtorrent

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment; refusing to start the ORFD GPU experiment")
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

bash scripts/download_efficient_sam_vits.sh
PYTHONPATH=src python - <<'PY'
from ttfm.model import build_model
build_model(2, "rod-vits", {
    "rod_encoder_checkpoint": "weights/efficient_sam_vits.pt",
    "rod_encoder_image_size": 1024,
})
build_model(2, "smp:FPN:efficientnet-b0", {"encoder_weights": "imagenet"})
print("ROD and FPN model preflight: PASS")
PY

if [[ "${ORFD_SKIP_DOWNLOAD:-0}" != "1" ]]; then
    python scripts/download_orfd_rgb.py --destination "$raw_root"
else
    echo "ORFD_SKIP_DOWNLOAD=1; using existing dataset below $raw_root"
fi
python scripts/prepare_orfd.py --raw-root "$raw_root" --processed-root data_processed_orfd

read -r -a seed_args <<< "$seeds"
PYTHONPATH=src python scripts/run_orfd_experiments.py --seeds "${seed_args[@]}"

echo "Complete. Read reports/orfd/comparison.md"
