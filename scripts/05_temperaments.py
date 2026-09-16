#!/usr/bin/env python
"""Chase vs. withdraw: does a losing fly get too sad to keep playing?

Sustained losses build a slow **despair state** (mood). With a "withdrawer"
temperament, deep sadness suppresses betting — the fly gives up and stops
(learned-helplessness-like), which can actually protect it from ruin. A
"chaser" keeps playing while sad and goes broke. Which one a fly is depends on
a single temperament parameter (`withdrawal`) — the same individual variation
you see across people.

    python scripts/05_temperaments.py            # one of each, makes the figure
    python scripts/05_temperaments.py --n 20     # + population: % ruin vs % gave-up

This is an in-silico model of an internal state, not a claim about anyone's
feelings; real gambling harm is a clinical matter with real help available.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygambler.connectome import load_tables, extract_circuit, mushroom_body_ids
from flygambler.fastlif import FastLIF
from flygambler.brain import FlyBrain, BrainConfig
from flygambler.game import RiggedBook
from flygambler.simulate import run_life, SimConfig, RUIN
from flygambler.plots import plot_temperaments

TEMPERAMENTS = {
    "chaser":     dict(withdrawal=0.0),
    "withdrawer": dict(withdrawal=10.0, mood_gain=0.5, mood_recovery=0.999),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--bankroll", type=float, default=25.0)
    ap.add_argument("--edge", type=float, default=0.06)
    ap.add_argument("--n", type=int, default=1, help="flies per temperament (>1 => population stats)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1_000_000)
    print(f">>> seed = {args.seed}")

    tables = load_tables(args.data)
    circuit = extract_circuit(tables, mushroom_body_ids(tables, side="right"), verbose=True)
    engine = FastLIF(circuit, dt_ms=0.5)
    cfg = SimConfig(bankroll0=args.bankroll, stake=1.0, target=100.0, max_rounds=args.rounds)

    def book_for(seed):
        return RiggedBook(n_channels=16, odds=2.0, active_channels=6, edge=args.edge, seed=seed)

    # one representative fly of each temperament (same book) -> the figure
    logs = {}
    for name, over in TEMPERAMENTS.items():
        brain = FlyBrain(None, BrainConfig(seed=0, simulate_dopamine=False, **over), engine=engine, verbose=False)
        brain.reset(seed=args.seed)
        log = run_life(brain, book_for(args.seed), cfg, verbose=False)
        logs[name] = log
        m = log.meta
        print(f"  {name:11s}: {m['outcome'].upper():7s} final ${m['final_bankroll']:.0f} "
              f"mood {m['final_mood']:+.2f} gave_up={m['gave_up']} bets={m['n_bets']}")
    fig = plot_temperaments(logs, ROOT / "runs" / "temperaments.png")
    print(f">>> figure: {fig}")

    # population: how often does each temperament ruin vs. give up?
    if args.n > 1:
        print(f"\n>>> population ({args.n} random flies per temperament):")
        for name, over in TEMPERAMENTS.items():
            brain = FlyBrain(None, BrainConfig(seed=0, simulate_dopamine=False, **over), engine=engine, verbose=False)
            brain.reset(seed=args.seed)
            brain.calibrate([book_for(args.seed).new_cue() for _ in range(40)])
            ruin = gave_up = 0
            finals = []
            for k in range(args.n):
                brain.reset(seed=args.seed + 7919 * k)
                m = run_life(brain, book_for(args.seed + 7919 * k), cfg, calibrate=False).meta
                ruin += m["outcome"] == RUIN
                gave_up += m["gave_up"]
                finals.append(m["final_bankroll"])
            print(f"  {name:11s}: ruined {100*ruin//args.n:3d}%   gave-up {100*gave_up//args.n:3d}%   "
                  f"mean final ${sum(finals)/len(finals):.1f}")


if __name__ == "__main__":
    main()
