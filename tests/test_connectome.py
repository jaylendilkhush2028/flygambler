"""Connectome-dependent checks. Skipped automatically if data/ isn't downloaded."""

from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
_have_data = (DATA / "Connectivity_783.parquet").exists() and (DATA / "neuron_annotations.tsv").exists()
pytestmark = pytest.mark.skipif(not _have_data, reason="connectome data not downloaded (run scripts/download_data.py)")


def test_mushroom_body_is_present_with_real_synapses():
    from flygambler.connectome import load_tables, mushroom_body_ids, extract_circuit
    tables = load_tables(DATA)
    circ = extract_circuit(tables, mushroom_body_ids(tables, side="right"), verbose=False)
    r = circ.role_idx
    assert len(r["KC"]) > 2000            # ~2,597 Kenyon cells (right hemisphere)
    assert len(r["MBON"]) >= 40           # ~48 MBONs
    assert len(r["PAM"]) > 100            # reward-dopamine cluster
    assert len(r["PPL1"]) >= 4            # punishment cluster
    assert len(r["ALPN"]) > 100           # cue input
    assert int(circ.is_plastic.sum()) > 5000   # real KC->MBON synapses (the learning substrate)


def test_fast_engine_produces_sparse_kc_code():
    from flygambler.connectome import load_tables, mushroom_body_ids, extract_circuit
    from flygambler.fastlif import FastLIF
    import numpy as np
    tables = load_tables(DATA)
    circ = extract_circuit(tables, mushroom_body_ids(tables, side="right"), verbose=False)
    eng = FastLIF(circ, dt_ms=0.5)
    alpn, kc = circ.role_idx["ALPN"], circ.role_idx["KC"]
    eng.reset_state()
    drive = np.zeros(circ.n); drive[alpn[:150]] = 45.0
    eng.set_stim_vector(drive)
    counts = eng.run_counts(100)
    frac = (counts[kc] > 0).mean()
    assert 0.01 < frac < 0.4              # a sparse (not empty, not saturated) Kenyon-cell code
