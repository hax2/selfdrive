from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": "selfdrive-dataset-downloader/1.0"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        resume = existing_size > 0 and response.status == 206
        mode = "ab" if resume else "wb"
        downloaded = existing_size if resume else 0
        response_size = int(response.headers.get("Content-Length", 0))
        total = downloaded + response_size if response_size else 0
        with destination.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\rDownloaded {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print(f"\nDownloaded: {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} URL DESTINATION")
    download(sys.argv[1], Path(sys.argv[2]))

