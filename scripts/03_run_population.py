#!/usr/bin/env python
"""Run a POPULATION of flies and see the distribution of fates.

Reuses one built network (fast). Shows how many go broke vs. get rich, and how
long ruin takes -- the population view of "how animals react to winning/losing."

    python scripts/03_run_population.py --book rigged --n 60
    python scripts/03_run_population.py --book learnable --n 60
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
from flygambler.simulate import run_population, SimConfig
from flygambler.plots import plot_population


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="rigged", choices=["learnable", "rigged"])
    ap.add_argument("--n", type=int, default=60, help="number of flies")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--bankroll", type=float, default=25.0)
    ap.add_argument("--stake", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--odds", type=float, default=2.0)
    ap.add_argument("--edge", type=float, default=0.06)
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--seed", type=int, default=None, help="base seed (random each run if omitted)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        import random
        args.seed = random.randrange(1_000_000)
    print(f">>> base seed = {args.seed} (each fly gets a different random seed)")

    print(">>> loading connectome ...")
    tables = load_tables(args.data)
    # population stats don't need the DAN-firing visualisation, so skip it for speed
    brain = make_brain(tables, BrainConfig(side=args.side, seed=0, simulate_dopamine=False))

    def book_factory(seed):
        kw = dict(n_channels=brain.cfg.n_channels, odds=args.odds, seed=seed)
        if args.book == "rigged":
            kw["edge"] = args.edge
        return make_book(args.book, **kw)

    cfg = SimConfig(bankroll0=args.bankroll, stake=args.stake,
                    target=args.target, max_rounds=args.rounds)
    print(f">>> running {args.n} flies on {args.book} book ...")
    pop = run_population(brain, book_factory, n_flies=args.n, cfg=cfg, base_seed=1000 + args.seed)

    df = pop.to_dataframe()
    n_ruin = int((df.outcome == "ruin").sum())
    n_rich = int((df.outcome == "rich").sum())
    n_to = int((df.outcome == "timeout").sum())
    print(f"\n=== {args.book} population (n={len(df)}) ===")
    print(f"    ruined (went broke): {n_ruin}  ({100*n_ruin/len(df):.0f}%)")
    print(f"    rich   (hit target): {n_rich}  ({100*n_rich/len(df):.0f}%)")
    print(f"    timeout            : {n_to}")
    print(f"    median final bankroll: ${df.final_bankroll.median():.1f}")
    if n_ruin:
        print(f"    median rounds to ruin: {int(df[df.outcome=='ruin'].n_rounds.median())}")

    out = Path(args.out) if args.out else ROOT / "runs" / f"pop_{args.book}.json"
    pop.save_json(out)
    print(f">>> saved: {out}")
    if not args.no_plot:
        fig = plot_population(pop, out.with_suffix(".png"),
                              title=f"{args.n} flies · {args.book} book")
        print(f">>> figure: {fig}")


if __name__ == "__main__":
    main()
