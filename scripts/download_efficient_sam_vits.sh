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

curl -L --fail --show-error \
    https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip \
    -o "$archive"
unzip -o "$archive" -d "$weights_dir"
rm "$archive"
echo "Downloaded: $checkpoint"
