"""Load the real FlyWire connectome and build a Brian2 spiking network from it.

The heavy lifting of turning the connectome dataframes into a leaky
integrate-and-fire (LIF) network follows the published whole-brain model of
Shiu et al. (Nature 2024) -- see ``vendor/drosophila_brain_model/model.py``.
We reuse their neuron equations and their rule for deriving synaptic weights
from synapse counts and neurotransmitter sign.

Two things we add on top of the published model:

1. **Cell-type roles.** We join the FlyWire annotations (Schlegel et al. 2024)
   so we can address the mushroom-body learning circuit by biological role:
   Kenyon cells (KC), mushroom-body output neurons (MBON), the reward
   dopamine cluster (PAM), the punishment dopamine cluster (PPL1), the APL
   feedback-inhibition neuron, and antennal-lobe projection neurons (ALPN)
   used as the cue-input pathway.

2. **A plastic learning substrate.** The KC->MBON synapses -- the real
   anatomical synapses where the fly stores learned valence -- are built as a
   *separate* Brian2 ``Synapses`` object so their weights can be rewritten
   between rounds by the dopamine-gated learning rule in ``brain.py``.
   (The published model has no plasticity; weights are fixed. This is our
   biologically-motivated extension.)

The same builder produces either a small mushroom-body **subnetwork** (a few
thousand real neurons -- fast enough to gamble live and to ruin/riches) or the
**full ~140k-neuron brain** (for validation / showcase), by choosing which
neuron IDs to include.
"""

from __future__ import annotations

import os as _os

# Activate setuptools' distutils shim before importing Brian2 (see __init__.py).
_os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
try:  # pragma: no cover
    import setuptools  # noqa: F401
except Exception:  # pragma: no cover
    pass

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from time import time

import numpy as np
import pandas as pd
from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    Network,
    mV,
    ms,
    volt,
    defaultclock,
)

# --------------------------------------------------------------------------
# LIF parameters (from Shiu et al. 2024, vendor/drosophila_brain_model/model.py)
# --------------------------------------------------------------------------
LIF_PARAMS = {
    "v_0": -52 * mV,     # resting potential      (Kakaria & de Bivort 2017)
    "v_rst": -52 * mV,   # reset after spike
    "v_th": -45 * mV,    # spike threshold
    "t_mbr": 20 * ms,    # membrane time constant
    "tau": 5 * ms,       # synaptic time constant (Juergensen et al. 2021)
    "t_rfc": 2.2 * ms,   # refractory period      (Lazar et al. 2021)
    "t_dly": 1.8 * ms,   # synaptic delay         (Paul et al. 2015)
    "w_syn": 0.275 * mV, # voltage bump per synapse (free parameter)
}

# Neuron equations. Identical to the published model plus an ``I_stim`` term:
# a per-neuron injected drive (in volts) we use to present the betting cue
# (on ALPNs) and to deliver dopamine (on PAM/PPL1), analogous to the
# optogenetic activation in the original paper but directly controllable
# round-to-round.
NEURON_EQS = dedent(
    """
    dv/dt = (v_0 - v + g + I_stim) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau                       : volt (unless refractory)
    I_stim                                 : volt
    rfc                                    : second
    """
)
THRESHOLD = "v > v_th"
RESET = "v = v_rst; g = 0*mV"

# Cell-type role -> selection predicate over the annotation table.
# Values are (column, kind, pattern).
ROLE_DEFS = {
    "KC":   ("cell_class", "eq", "Kenyon_Cell"),
    "MBON": ("cell_class", "eq", "MBON"),
    "PAM":  ("cell_type", "startswith", "PAM"),   # reward dopamine cluster
    "PPL1": ("cell_type", "startswith", "PPL1"),  # punishment dopamine cluster
    "APL":  ("cell_type", "fullmatch", "APL"),    # feedback inhibition
    "ALPN": ("cell_class", "eq", "ALPN"),         # antennal-lobe projection neurons (cue input)
}


@dataclass
class ConnectomeTables:
    """Raw connectome + annotations, loaded once and reused."""

    comp: pd.DataFrame           # index = flywire_id, column 'Completed'
    con: pd.DataFrame            # edge list with IDs, synapse count, sign
    ann: pd.DataFrame            # annotations, index = root_id
    data_dir: Path

    @property
    def all_ids(self) -> np.ndarray:
        return self.comp.index.to_numpy(dtype=np.int64)

    def role_ids(self, role: str, side: str | None = "right") -> np.ndarray:
        """flywire IDs for a biological role, optionally restricted to a hemisphere."""
        col, kind, pat = ROLE_DEFS[role]
        s = self.ann[col].astype(str)
        if kind == "eq":
            mask = s.eq(pat)
        elif kind == "startswith":
            mask = s.str.startswith(pat)
        elif kind == "fullmatch":
            mask = s.str.fullmatch(pat)
        else:  # pragma: no cover
            raise ValueError(kind)
        if side is not None:
            mask &= self.ann["side"].astype(str).eq(side)
        ids = self.ann.loc[mask, "root_id"].to_numpy(dtype=np.int64)
        # keep only neurons that actually exist in the connectome
        in_conn = np.intersect1d(ids, self.all_ids)
        return in_conn


def load_tables(data_dir: str | Path) -> ConnectomeTables:
    """Load the connectome dataframes and the FlyWire annotations."""
    data_dir = Path(data_dir)
    comp = pd.read_csv(data_dir / "Completeness_783.csv", index_col=0)
    comp.index = comp.index.astype(np.int64)
    con = pd.read_parquet(
        data_dir / "Connectivity_783.parquet",
        columns=[
            "Presynaptic_ID",
            "Postsynaptic_ID",
            "Connectivity",
            "Excitatory x Connectivity",
        ],
    )
    ann = pd.read_csv(data_dir / "neuron_annotations.tsv", sep="\t", low_memory=False)
    ann["root_id"] = ann["root_id"].astype(np.int64)
    return ConnectomeTables(comp=comp, con=con, ann=ann, data_dir=data_dir)


@dataclass
class Circuit:
    """Engine-agnostic description of a connectome subcircuit.

    Local neuron indices run 0..n-1. Weights are signed and expressed in **mV**
    (``Excitatory x Connectivity`` * ``w_syn``); the sign carries excitatory(+)
    vs inhibitory(-) from the neurotransmitter prediction. Both the Brian2
    builder and the fast NumPy engine consume this identical description.
    """

    n: int
    pre_i: np.ndarray            # presynaptic local index per edge
    post_i: np.ndarray           # postsynaptic local index per edge
    w_mV: np.ndarray             # signed weight per edge, in mV
    is_plastic: np.ndarray       # bool mask: KC->MBON edges (the learning substrate)
    role_idx: dict               # role -> local neuron indices
    id2i: dict
    i2id: dict
    params: dict = field(default_factory=lambda: dict(LIF_PARAMS))

    @property
    def plastic_pre_kc(self) -> np.ndarray:
        return self.pre_i[self.is_plastic]

    @property
    def plastic_post_mbon(self) -> np.ndarray:
        return self.post_i[self.is_plastic]

    @property
    def w0_plastic(self) -> np.ndarray:
        return self.w_mV[self.is_plastic].astype(np.float64)


def extract_circuit(
    tables: ConnectomeTables,
    include_ids: np.ndarray,
    *,
    params: dict | None = None,
    verbose: bool = True,
) -> Circuit:
    """Filter the connectome to ``include_ids`` and return a :class:`Circuit`.

    KC->MBON edges among the included neurons are flagged plastic. This is the
    single source of truth for wiring + weights shared by both engines.
    """
    params = dict(LIF_PARAMS if params is None else params)
    include_ids = np.unique(np.asarray(include_ids, dtype=np.int64))
    n = len(include_ids)

    id2i = {int(fid): i for i, fid in enumerate(include_ids)}
    i2id = {i: int(fid) for fid, i in id2i.items()}
    include_set = set(id2i)

    t0 = time()
    con = tables.con
    inc_index = pd.Index(include_ids)
    id2i_ser = pd.Series(np.arange(n, dtype=np.int64), index=include_ids)

    mask = con["Presynaptic_ID"].isin(inc_index) & con["Postsynaptic_ID"].isin(inc_index)
    sub = con.loc[mask]
    if verbose:
        print(f"    edges within subset: {len(sub):,} (of {len(con):,})  [{time()-t0:.1f}s]")

    pre_ids = sub["Presynaptic_ID"].to_numpy(np.int64)
    post_ids = sub["Postsynaptic_ID"].to_numpy(np.int64)
    pre_i = sub["Presynaptic_ID"].map(id2i_ser).to_numpy(np.int64)
    post_i = sub["Postsynaptic_ID"].map(id2i_ser).to_numpy(np.int64)
    w_mV = sub["Excitatory x Connectivity"].to_numpy(np.float64) * float(params["w_syn"] / mV)

    role_ids_fw: dict[str, np.ndarray] = {}
    role_idx: dict[str, np.ndarray] = {}
    for role in ROLE_DEFS:
        fw = np.array([fid for fid in tables.role_ids(role, side=None) if int(fid) in include_set], dtype=np.int64)
        role_ids_fw[role] = fw
        role_idx[role] = np.sort(id2i_ser.reindex(fw).to_numpy(np.int64))

    kc_fw = pd.Index(role_ids_fw["KC"])
    mbon_fw = pd.Index(role_ids_fw["MBON"])
    is_plastic = (pd.Index(pre_ids).isin(kc_fw) & pd.Index(post_ids).isin(mbon_fw))
    is_plastic = np.asarray(is_plastic, dtype=bool)

    if verbose:
        print(
            f"    circuit: {n:,} neurons | {int((~is_plastic).sum()):,} static + "
            f"{int(is_plastic.sum()):,} plastic (KC->MBON) edges  [{time()-t0:.1f}s]"
        )
        for role in ("KC", "MBON", "PAM", "PPL1", "APL", "ALPN"):
            print(f"      {role:5s}: {len(role_idx[role]):5d}")

    return Circuit(
        n=n, pre_i=pre_i, post_i=post_i, w_mV=w_mV, is_plastic=is_plastic,
        role_idx=role_idx, id2i=id2i, i2id=i2id, params=params,
    )


@dataclass
class BrainScaffold:
    """A built Brian2 network plus the handles needed to run rounds (used for
    whole-brain validation; the interactive loop uses the fast NumPy engine)."""

    net: Network
    neu: NeuronGroup
    syn_static: Synapses
    syn_plastic: Synapses
    spk_mon: SpikeMonitor
    circuit: Circuit

    @property
    def role_idx(self):
        return self.circuit.role_idx

    @property
    def params(self):
        return self.circuit.params

    @property
    def n_neurons(self) -> int:
        return len(self.neu)


def build_scaffold(
    tables: ConnectomeTables,
    include_ids: np.ndarray | None = None,
    *,
    circuit: Circuit | None = None,
    params: dict | None = None,
    record_spikes: bool = False,
    verbose: bool = True,
) -> BrainScaffold:
    """Build a Brian2 LIF network from a :class:`Circuit` (or by extracting one)."""
    if circuit is None:
        circuit = extract_circuit(tables, include_ids, params=params, verbose=verbose)
    params = circuit.params
    n = circuit.n

    neu = NeuronGroup(
        n, model=NEURON_EQS, method="euler", threshold=THRESHOLD, reset=RESET,
        refractory="rfc", name="fly_neurons", namespace=params,
    )
    neu.v = params["v_0"]
    neu.g = 0 * mV
    neu.I_stim = 0 * mV
    neu.rfc = params["t_rfc"]

    pl = circuit.is_plastic
    syn_static = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params["t_dly"], name="static_synapses")
    syn_static.connect(i=circuit.pre_i[~pl], j=circuit.post_i[~pl])
    syn_static.w = circuit.w_mV[~pl] * mV

    syn_plastic = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=params["t_dly"], name="plastic_KC_MBON")
    syn_plastic.connect(i=circuit.pre_i[pl], j=circuit.post_i[pl])
    syn_plastic.w = circuit.w_mV[pl] * mV

    spk_mon = SpikeMonitor(neu, record=record_spikes)
    net = Network(neu, syn_static, syn_plastic, spk_mon)

    return BrainScaffold(net=net, neu=neu, syn_static=syn_static, syn_plastic=syn_plastic,
                         spk_mon=spk_mon, circuit=circuit)


def mushroom_body_ids(tables: ConnectomeTables, side: str | None = "right") -> np.ndarray:
    """flywire IDs of the mushroom-body learning ecosystem (one hemisphere by default)."""
    roles = ["KC", "MBON", "PAM", "PPL1", "APL", "ALPN"]
    ids = np.concatenate([tables.role_ids(r, side=side) for r in roles])
    return np.unique(ids)
