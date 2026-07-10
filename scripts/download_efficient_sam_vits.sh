#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
weights_dir="$root_dir/weights"
archive="$weights_dir/efficient_sam_vits.pt.zip"
checkpoint="$weights_dir/efficient_sam_vits.pt"

mkdir -p "$weights_dir"
if [[ -f "$checkpoint" ]]; then
    echo "Checkpoint already exists: $checkpoint"
    exit 0
fi

url="https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --show-error "$url" -o "$archive"
else
    python "$root_dir/scripts/download_file.py" "$url" "$archive"
fi
if command -v unzip >/dev/null 2>&1; then
    unzip -o "$archive" -d "$weights_dir"
else
    python -m zipfile -e "$archive" "$weights_dir"
fi
rm "$archive"
echo "Downloaded: $checkpoint"
