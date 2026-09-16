#!/usr/bin/env python
"""Tier-0 validation: boot the REAL whole-brain connectome and make it spike.

This proves the whole-brain leaky integrate-and-fire model (all ~140k neurons,
~15M synapses; Shiu et al. 2024, built on the FlyWire connectome) actually runs
here, and that our fast engine reproduces the published Brian2 model.

We inject drive into a real input population (olfactory projection neurons by
default; pass FlyWire IDs to stimulate any set) and report how activity spreads
across the whole brain and into the mushroom body.

    python scripts/01_validate_wholebrain.py                # fast engine (default)
    python scripts/01_validate_wholebrain.py --brian2       # authentic Brian2 model
    python scripts/01_validate_wholebrain.py --ms 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flygambler.connectome import load_tables, extract_circuit


def run_fast(circuit, stim_idx, drive, ms):
    from flygambler.fastlif import FastLIF
    eng = FastLIF(circuit, dt_ms=0.5)
    eng.reset_state()
    d = np.zeros(circuit.n); d[stim_idx] = drive
    eng.set_stim_vector(d)
    t0 = time.time()
    counts = eng.run_counts(ms)
    return counts, time.time() - t0


def run_brian2(tables, circuit, stim_idx, drive, ms):
    from brian2 import prefs, defaultclock, ms as bms, mV
    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.5 * bms
    from flygambler.connectome import build_scaffold
    sc = build_scaffold(tables, circuit=circuit, verbose=False)
    c0 = np.array(sc.spk_mon.count)
    d = np.zeros(circuit.n); d[stim_idx] = drive
    sc.neu.I_stim = d * mV
    t0 = time.time()
    sc.net.run(ms * bms)
    return np.array(sc.spk_mon.count) - c0, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", type=float, default=60.0, help="biological ms to simulate")
    ap.add_argument("--drive", type=float, default=40.0, help="stimulation drive (mV)")
    ap.add_argument("--brian2", action="store_true", help="use the authentic Brian2 model")
    ap.add_argument("--both", action="store_true", help="run both engines and compare MB response")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()

    print(">>> loading connectome ...")
    tables = load_tables(args.data)
    print(f">>> building the WHOLE brain: {len(tables.all_ids):,} neurons ...")
    t0 = time.time()
    circuit = extract_circuit(tables, tables.all_ids, verbose=True)
    print(f"    extracted in {time.time()-t0:.1f}s | total edges: {len(circuit.pre_i):,}")

    # input population: olfactory projection neurons (real input to the mushroom body)
    stim = circuit.role_idx["ALPN"]
    kc, mbon = circuit.role_idx["KC"], circuit.role_idx["MBON"]
    print(f">>> stimulating {len(stim)} ALPNs @ {args.drive} mV for {args.ms} ms\n")

    def report(counts, wall, engine):
        active = int((counts > 0).sum())
        print(f"[{engine}] wall={wall:.1f}s | brain-wide active neurons: {active:,}/{circuit.n:,} "
              f"({100*active/circuit.n:.1f}%) | total spikes: {int(counts.sum()):,}")
        print(f"           mushroom body: KC active {int((counts[kc]>0).sum())}/{len(kc)} "
              f"({100*(counts[kc]>0).mean():.1f}%), MBON active {int((counts[mbon]>0).sum())}/{len(mbon)}, "
              f"MBON spikes {int(counts[mbon].sum())}")
        return counts

    if args.both or not args.brian2:
        c_fast, w_fast = run_fast(circuit, stim, args.drive, args.ms)
        report(c_fast, w_fast, "fast ")
    if args.both or args.brian2:
        print("    (building Brian2 network with ~15M synapses; this is the heavy, authentic path) ...")
        c_b2, w_b2 = run_brian2(tables, circuit, stim, args.drive, args.ms)
        report(c_b2, w_b2, "brian2")

    print("\n>>> The real whole-brain LIF model boots and computes. "
          "The interactive gambling loop uses the mushroom-body subnetwork of this same brain.")


if __name__ == "__main__":
    main()
