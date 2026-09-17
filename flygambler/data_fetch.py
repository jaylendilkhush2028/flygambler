"""Download the connectome data this project needs into ``data/``.

Run ``python scripts/download_data.py`` (or ``flygambler-fetch-data`` if
installed). Files are skipped if already present; pass ``--force`` to re-fetch.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (filename in data/, source URL). All public, no login required.
FILES = [
    ("Completeness_783.csv",
     "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv"),
    ("Connectivity_783.parquet",
     "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet"),
    ("neuron_annotations.tsv",
     "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"),
]


def _download(url: str, dest: Path) -> None:
    def hook(block, block_size, total):
        if total and total > 0:
            pct = min(100.0, 100.0 * block * block_size / total)
            sys.stdout.write(f"\r    {dest.name}: {pct:5.1f}%  ({block * block_size // 1_000_000} MB)")
            sys.stdout.flush()
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    sys.stdout.write("\n")


def main(data_dir: str | Path | None = None, force: bool = False) -> None:
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> fetching connectome data into {data_dir}/  (~135 MB total, one time)")
    for name, url in FILES:
        dest = data_dir / name
        if dest.exists() and not force:
            print(f"    {name}: already present ({dest.stat().st_size // 1_000_000} MB) — skipping")
            continue
        print(f"    downloading {name} ...")
        _download(url, dest)
    print(">>> done. Now run e.g.  python scripts/02_run_fly.py --book learnable")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
