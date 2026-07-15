#!/usr/bin/env python3
"""Download only the RGB images and ground-truth masks from the ORFD torrent."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path


TORRENT_URL = "https://academictorrents.com/download/ec5ccf4b8e49271ee3b63660383facf43063f2f2.torrent"


def _download_torrent_file(destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(TORRENT_URL, headers={"User-Agent": "ttfm-orfd/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("datasets/ORFD"))
    parser.add_argument("--torrent", type=Path, default=Path("downloads/orfd.torrent"))
    parser.add_argument(
        "--stall-timeout",
        type=int,
        default=1800,
        help="Abort after this many seconds with no downloaded bytes (default: 1800)",
    )
    args = parser.parse_args()

    try:
        import libtorrent as lt
    except ImportError as error:
        raise SystemExit(
            "The ORFD downloader needs the libtorrent Python wheel. "
            "Run it through scripts/run_orfd_comparison.sh, which installs it automatically."
        ) from error

    _download_torrent_file(args.torrent)
    args.destination.mkdir(parents=True, exist_ok=True)

    info = lt.torrent_info(str(args.torrent))
    priorities: list[int] = []
    selected_bytes = 0
    selected_files = 0
    for index in range(info.files().num_files()):
        path = info.files().file_path(index).replace("\\", "/")
        wanted = "/image_data/" in f"/{path}" or "/gt_image/" in f"/{path}"
        priorities.append(4 if wanted else 0)
        if wanted:
            selected_files += 1
            selected_bytes += info.files().file_size(index)
    if selected_files == 0:
        raise RuntimeError("The ORFD torrent contained no image_data or gt_image files")

    print(
        f"Selected {selected_files:,} RGB/label files ({selected_bytes / 1e9:.2f} GB) "
        f"from the {info.total_size() / 1e9:.2f} GB ORFD torrent.",
        flush=True,
    )
    session = lt.session({"listen_interfaces": "0.0.0.0:6881", "enable_dht": True})
    handle = session.add_torrent(
        {
            "ti": info,
            "save_path": str(args.destination.resolve()),
            "storage_mode": lt.storage_mode_t.storage_mode_sparse,
        }
    )
    handle.prioritize_files(priorities)

    last_line_length = 0
    last_progress = -1
    last_progress_time = time.monotonic()
    while True:
        status = handle.status()
        wanted_done = status.total_wanted_done
        wanted_total = max(status.total_wanted, 1)
        line = (
            f"ORFD RGB/labels: {wanted_done / 1e9:.2f}/{wanted_total / 1e9:.2f} GB "
            f"({wanted_done / wanted_total * 100:5.1f}%) "
            f"down={status.download_rate / 1e6:.1f} MB/s peers={status.num_peers}"
        )
        print("\r" + line.ljust(last_line_length), end="", flush=True)
        last_line_length = len(line)
        if status.is_finished:
            break
        if wanted_done != last_progress:
            last_progress = wanted_done
            last_progress_time = time.monotonic()
        elif time.monotonic() - last_progress_time > args.stall_timeout:
            raise TimeoutError(
                f"ORFD download made no progress for {args.stall_timeout} seconds; "
                "rerun later to resume, or check firewall/BitTorrent access"
            )
        if status.errc.value() != 0:
            raise RuntimeError(f"ORFD torrent error: {status.errc.message()}")
        time.sleep(2)
    print("\nORFD RGB and ground-truth download complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; rerun the command to verify and resume the download.", file=sys.stderr)
        raise SystemExit(130)
