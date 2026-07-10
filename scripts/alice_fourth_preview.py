from __future__ import annotations

import argparse
from pathlib import Path

from ttfm.utils import load_yaml

from alice_preview import main as alice_preview_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sampled alice previews with the fourth-run checkpoint.")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--input-dir", default="alice")
    parser.add_argument("--output-dir", default="outputs/alice_fourth_preview")
    args = parser.parse_args()

    # Reinvoke the generic preview script defaults with the fourth-run config.
    import sys

    sys.argv = [
        "alice_preview.py",
        "--config",
        "configs/fourth_run.yaml",
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--sample-count",
        str(args.sample_count),
        "--checkpoint",
        "best.pt",
    ]
    alice_preview_main()


if __name__ == "__main__":
    main()
