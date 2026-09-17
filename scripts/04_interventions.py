#!/usr/bin/env python
"""Innate vs. learned, and 'how do you stop the pain?' — an intervention study.

The pain of losing is *innate* here (the PPL1 punishment neurons fire on every
loss regardless of anything). But the *compulsive loss-chasing that leads to
ruin is learned* — it lives in the dopamine-gated KC->MBON plasticity. So we
run populations of RANDOM flies on the rigged (unbeatable) book under different
interventions and ask which ones reduce ruin and total accumulated pain:

  * Baseline          — a normal fly.
  * No learning       — plasticity off (isolates the innate reaction).
  * Reward-blocked    — wins no longer reinforce betting (kills the "chase").
  * Fast forgetting   — strong extinction back toward baseline.
  * Self-control      — a higher decision threshold (bets more reluctantly).

This is an in-silico mechanistic experiment, not medical advice: it shows which
*mechanisms* drive the harm in the model, which is the useful, testable output.

    python scripts/04_interventions.py --n 10
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygambler.connectome import load_tables, extract_circuit, mushroom_body_ids
from flygambler.fastlif import FastLIF
from flygambler.brain import FlyBrain, BrainConfig
from flygambler.game import RiggedBook
from flygambler.simulate import run_life, SimConfig, RUIN, RICH

CONDITIONS = [
    ("Baseline", {}),
    ("No learning", {"learn_rate": 0.0}),
    ("Reward-blocked", {"reward_scale": 0.0}),
    ("Fast forgetting", {"decay": 0.03}),
    ("Self-control", {"bet_bias": 0.9}),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="flies per condition")
    ap.add_argument("--rounds", type=int, default=350)
    ap.add_argument("--bankroll", type=float, default=25.0)
    ap.add_argument("--target", type=float, default=100.0)
    ap.add_argument("--edge", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1_000_000)
    print(f">>> base seed = {args.seed}")

    print(">>> loading connectome + building mushroom-body engine (shared) ...")
    tables = load_tables(args.data)
    circuit = extract_circuit(tables, mushroom_body_ids(tables, side="right"), verbose=True)
    engine = FastLIF(circuit, dt_ms=0.5)
    cfg = SimConfig(bankroll0=args.bankroll, stake=1.0, target=args.target, max_rounds=args.rounds)

    results = []
    for ci, (name, over) in enumerate(CONDITIONS):
        bcfg = BrainConfig(seed=0, simulate_dopamine=False, **over)
        brain = FlyBrain(None, bcfg, engine=engine, verbose=False)
        # calibrate once on this book family
        brain.reset(seed=args.seed)
        brain.calibrate([RiggedBook(n_channels=bcfg.n_channels, edge=args.edge, seed=args.seed).new_cue()
                         for _ in range(40)])
        fates, rounds_, pain, final = [], [], [], []
        t0 = time.time()
        for k in range(args.n):
            s = args.seed + 101 * k + 100_000 * ci
            brain.reset(seed=s)
            book = RiggedBook(n_channels=bcfg.n_channels, odds=2.0, active_channels=bcfg.cue_active_channels,
                              edge=args.edge, seed=s)
            log = run_life(brain, book, cfg, calibrate=False)
            m = log.meta
            fates.append(m["outcome"]); rounds_.append(m["n_rounds"])
            pain.append(m["cum_hurt"]); final.append(m["final_bankroll"])
        res = {
            "name": name,
            "ruin_rate": sum(f == RUIN for f in fates) / len(fates),
            "rich_rate": sum(f == RICH for f in fates) / len(fates),
            "median_survival": st.median(rounds_),
            "mean_pain": st.mean(pain),
            "pain_sem": (st.stdev(pain) / (len(pain) ** 0.5)) if len(pain) > 1 else 0.0,
            "mean_final": st.mean(final),
            "n": args.n,
        }
        results.append(res)
        print(f"  {name:16s}: ruin {100*res['ruin_rate']:3.0f}%  "
              f"median-survival {res['median_survival']:4.0f}  mean-pain {res['mean_pain']:6.1f}  "
              f"mean-final ${res['mean_final']:5.1f}  [{time.time()-t0:.0f}s]")

    out = ROOT / "runs" / "interventions.json"
    out.write_text(json.dumps(results, indent=2))
    from flygambler.plots import plot_interventions
    fig = plot_interventions(results, ROOT / "runs" / "interventions.png",
                             title=f"Reducing gambling harm in-silico · rigged book · {args.n} flies/condition")
    print(f">>> saved {out} and {fig}")

    base = results[0]
    print("\n=== takeaways ===")
    print(f"  Baseline flies ruin {100*base['ruin_rate']:.0f}% of the time.")
    for r in results[1:]:
        d = 100 * (base["ruin_rate"] - r["ruin_rate"])
        dp = 100 * (base["mean_pain"] - r["mean_pain"]) / max(1e-9, base["mean_pain"])
        print(f"  {r['name']:16s}: ruin {'-' if d>=0 else '+'}{abs(d):.0f} pts, "
              f"pain {'-' if dp>=0 else '+'}{abs(dp):.0f}% vs baseline")


if __name__ == "__main__":
    main()
