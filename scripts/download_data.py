#!/usr/bin/env python
"""Fetch the connectome data into data/ (one time). See flygambler/data_fetch.py.

    python scripts/download_data.py            # download what's missing
    python scripts/download_data.py --force    # re-download everything
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygambler.data_fetch import main

if __name__ == "__main__":
    main(force="--force" in sys.argv)
