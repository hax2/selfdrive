#!/usr/bin/env python3
from pathlib import Path
import subprocess
import zipfile

output = Path("rod_blue_results.zip")

files = [
    Path("outputs/mixed_binary_traversability_rod_vits_blue/checkpoints/best.pt"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/history.json"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/train_summary.json"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/test_metrics.json"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/test_review/summary.json"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/test_review/summary.md"),
    Path("configs/rod_vits_cat_blue.yaml"),
    Path("rod_blue_full.log"),
]

directories = [
    Path("outputs/mixed_binary_traversability_rod_vits_blue/visuals"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/test_visuals"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/test_review/worst_false_safe"),
    Path("outputs/mixed_binary_traversability_rod_vits_blue/test_review/worst_false_block"),
]

missing = [str(path) for path in files if not path.exists()]
if missing:
    print(f"Warning: Missing files: {missing}. Proceeding without them.")

with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
    for path in files:
        if path.exists():
            archive.write(path, path.as_posix())

    for directory in directories:
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, path.as_posix())

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        archive.writestr("RUN_COMMIT.txt", commit + "\n")
    except subprocess.CalledProcessError:
        pass

print(f"Created {output}: {output.stat().st_size / 1024**2:.1f} MiB")
