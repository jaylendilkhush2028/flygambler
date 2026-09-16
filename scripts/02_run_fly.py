#!/usr/bin/env python
"""Run ONE fly's gambling life on the real mushroom-body circuit, to ruin or riches.

Writes a run-log (JSON, for the dashboard) and a life-story figure.

    python scripts/02_run_fly.py --book learnable
    python scripts/02_run_fly.py --book rigged --rounds 2000 --seed 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygambler.connectome import load_tables
from flygambler.brain import make_brain, BrainConfig
from flygambler.game import make_book
from flygambler.simulate import run_life, SimConfig
from flygambler.plots import plot_life


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="learnable", choices=["learnable", "rigged"])
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--bankroll", type=float, default=25.0)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--odds", type=float, default=2.0)
    ap.add_argument("--edge", type=float, default=0.06, help="house edge for the rigged book")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--seed", type=int, default=None, help="fly seed (random each run if omitted)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="run-log JSON path")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        import random
        args.seed = random.randrange(1_000_000)
    print(f">>> fly seed = {args.seed}")

    print(">>> loading connectome ...")
    tables = load_tables(args.data)
    # fixed network structure (seed 0); the fly's behaviour varies with --seed
    brain = make_brain(tables, BrainConfig(side=args.side, seed=0))
    brain.reset(seed=args.seed)

    book_kw = dict(n_channels=brain.cfg.n_channels, odds=args.odds, seed=7)
    if args.book == "rigged":
        book_kw["edge"] = args.edge
    book = make_book(args.book, **book_kw)
    print(f">>> book={type(book).__name__} odds={args.odds} house_edge≈{book.house_edge*100:.1f}%")

    cfg = SimConfig(bankroll0=args.bankroll, stake=args.stake, target=args.target, max_rounds=args.rounds)
    print(f">>> gambling (up to {args.rounds} rounds) ...")
    log = run_life(brain, book, cfg, verbose=True)

    out = Path(args.out) if args.out else ROOT / "runs" / f"life_{args.book}_seed{args.seed}.json"
    log.save_json(out)
    print(f">>> run-log: {out}")
    if not args.no_plot:
        fig = plot_life(log, out.with_suffix(".png"))
        print(f">>> figure:  {fig}")


if __name__ == "__main__":
    main()
