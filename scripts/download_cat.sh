#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="$root_dir/CaT.tar.gz"
dataset_dir="$root_dir/CAT"
url="https://www.cavs.msstate.edu/resources/downloads/CaT/CaT.tar.gz"

if [[ -d "$dataset_dir/mixed/Train/imgs" && -d "$dataset_dir/mixed/Test/imgs" ]]; then
    echo "CaT dataset already exists: $dataset_dir"
    exit 0
fi

if command -v curl >/dev/null 2>&1; then
    curl -L --fail --show-error --continue-at - "$url" -o "$archive"
else
    python "$root_dir/scripts/download_file.py" "$url" "$archive"
fi

extract_dir="$(mktemp -d "$root_dir/.cat_extract.XXXXXX")"
trap 'rm -rf "$extract_dir"' EXIT
tar -xzf "$archive" -C "$extract_dir"

source_dir=""
for candidate in "$extract_dir/CAT" "$extract_dir/CaT" "$extract_dir/cat"; do
    if [[ -d "$candidate/mixed/Train/imgs" ]]; then
        source_dir="$candidate"
        break
    fi
done
if [[ -z "$source_dir" && -d "$extract_dir/mixed/Train/imgs" ]]; then
    source_dir="$extract_dir"
fi
if [[ -z "$source_dir" ]]; then
    echo "Downloaded archive does not contain the expected CaT layout." >&2
    exit 1
fi

rm -rf "$dataset_dir"
mkdir -p "$dataset_dir"
cp -a "$source_dir"/. "$dataset_dir"/
rm -f "$archive"
echo "Downloaded and extracted CaT: $dataset_dir"
