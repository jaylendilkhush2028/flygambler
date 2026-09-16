#!/usr/bin/env python
"""Watch the REAL fly-brain connectome gamble FOREVER, live in your terminal.

It plays endless lives on the real mushroom-body network: each time it hits $0
(broke) or $100 (rich) a new fly is born, and every life is archived to
runs/lives.jsonl. Press Ctrl+C to stop.

    python scripts/06_watch_forever.py                       # rigged + chaser
    python scripts/06_watch_forever.py --temperament withdrawer
    python scripts/06_watch_forever.py --book learnable --target 100
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygambler.connectome import load_tables
from flygambler.brain import make_brain, BrainConfig
from flygambler.game import make_book
from flygambler.simulate import SimConfig
from flygambler.live_view import watch_forever

TEMPERAMENTS = {
    "chaser": {},
    "withdrawer": {"withdrawal": 10.0, "mood_gain": 0.5, "mood_recovery": 0.999},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="rigged", choices=["rigged", "learnable"])
    ap.add_argument("--temperament", default="chaser", choices=["chaser", "withdrawer"])
    ap.add_argument("--bankroll", type=float, default=25.0)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--edge", type=float, default=0.06)
    ap.add_argument("--per-life-cap", type=int, default=4000,
                    help="max rounds before a life times out (so it keeps cycling); 0 = no cap")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--lives", default=str(ROOT / "runs" / "lives.jsonl"))
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1_000_000)

    print(">>> loading the real connectome + building the mushroom-body network ...")
    tables = load_tables(args.data)
    bcfg = BrainConfig(seed=0, simulate_dopamine=True, **TEMPERAMENTS[args.temperament])
    brain = make_brain(tables, bcfg, verbose=True)

    def book_factory(seed):
        kw = dict(n_channels=bcfg.n_channels, odds=2.0, active_channels=bcfg.cue_active_channels, seed=seed)
        if args.book == "rigged":
            kw["edge"] = args.edge
        return make_book(args.book, **kw)

    cfg = SimConfig(bankroll0=args.bankroll, stake=1.0, target=args.target,
                    max_rounds=(None if args.per_life_cap == 0 else args.per_life_cap))
    print(f">>> watching {args.temperament} on the {args.book} book — forever (Ctrl+C to stop)\n")
    watch_forever(brain, book_factory, cfg, base_seed=args.seed, lives_path=args.lives,
                  temperament=args.temperament, book_name=args.book)


if __name__ == "__main__":
    main()
