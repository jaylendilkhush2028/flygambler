"""A lightweight NumPy/SciPy integrator of the *same* LIF model, for speed.

Brian2 is excellent but pays a large fixed cost per ``run()`` call (network
prep), which is fine for a few long whole-brain runs but crippling for a
gambling loop that needs *thousands* of short runs. This module integrates the
identical equations (same constants, same connectome weights and signs) with a
plain Euler step over sparse weight matrices, eliminating that overhead.

Equations (identical to ``connectome.NEURON_EQS`` / Shiu et al. 2024):

    dv/dt = (v_0 - v + g + I_stim) / t_mbr
    dg/dt = -g / tau
    spike when v > v_th  ->  v <- v_rst, g <- 0, refractory for t_rfc
    on presynaptic spike (after delay t_dly): g_post += w

``FastLIF`` is validated against Brian2 in ``scripts/00_validate_engine`` /
the test suite: same cue -> matching KC sparseness and MBON drive.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from brian2 import mV, ms

from .connectome import Circuit


class FastLIF:
    """Fast Euler LIF engine over a :class:`Circuit`. Units: mV and ms."""

    def __init__(self, circuit: Circuit, dt_ms: float = 0.5):
        self.circuit = circuit
        self.role_idx = circuit.role_idx
        self.n = int(circuit.n)
        self.n_neurons = self.n

        p = circuit.params
        self.dt = float(dt_ms)
        self.v_0 = float(p["v_0"] / mV)
        self.v_rst = float(p["v_rst"] / mV)
        self.v_th = float(p["v_th"] / mV)
        self.t_mbr = float(p["t_mbr"] / ms)
        self.tau = float(p["tau"] / ms)
        self.refr_steps = int(round(float(p["t_rfc"] / ms) / self.dt))
        self.delay_steps = max(1, int(round(float(p["t_dly"] / ms) / self.dt)))

        pl = circuit.is_plastic
        # static synapses as CSR (row = postsynaptic, col = presynaptic)
        s_pre, s_post, s_w = circuit.pre_i[~pl], circuit.post_i[~pl], circuit.w_mV[~pl]
        self.W_static = sparse.csr_matrix(
            (s_w.astype(np.float64), (s_post, s_pre)), shape=(self.n, self.n)
        )
        # plastic synapses (KC->MBON) kept as editable arrays
        self.pl_pre = circuit.pre_i[pl].astype(np.int64)     # presynaptic KC (local idx)
        self.pl_post = circuit.post_i[pl].astype(np.int64)   # postsynaptic MBON (local idx)
        self.w_plastic = circuit.w_mV[pl].astype(np.float64).copy()
        self.plastic_pre_kc = self.pl_pre
        self.plastic_post_mbon = self.pl_post
        self.w0_plastic = circuit.w_mV[pl].astype(np.float64).copy()

        # state
        self.v = np.full(self.n, self.v_0)
        self.g = np.zeros(self.n)
        self.I_stim = np.zeros(self.n)
        self.refr = np.zeros(self.n, dtype=np.int32)
        self._ring = np.zeros((self.delay_steps, self.n), dtype=np.float64)
        self._head = 0
        self.reset_state()

    # --------------------------------------------------------------- state I/O
    def reset_state(self) -> None:
        self.v[:] = self.v_0
        self.g[:] = 0.0
        self.I_stim[:] = 0.0
        self.refr[:] = 0
        self._ring[:] = 0.0
        self._head = 0

    def set_stim_vector(self, drive_mV: np.ndarray) -> None:
        self.I_stim[:] = drive_mV

    def set_stim(self, idx, drive_mV) -> None:
        self.I_stim[:] = 0.0
        self.I_stim[idx] = drive_mV

    def get_plastic(self) -> np.ndarray:
        return self.w_plastic

    def set_plastic(self, w_mV: np.ndarray) -> None:
        self.w_plastic[:] = w_mV

    # ------------------------------------------------------------------- solve
    def _step(self) -> np.ndarray:
        active = self.refr <= 0
        # integrate membrane + conductance (frozen while refractory)
        dv = self.dt * (self.v_0 - self.v + self.g + self.I_stim) / self.t_mbr
        dg = -self.dt * self.g / self.tau
        self.v[active] += dv[active]
        self.g[active] += dg[active]

        # deliver spikes emitted delay_steps ago
        arrivals = self._ring[self._head]
        if arrivals.any():
            ginc = self.W_static.dot(arrivals)
            # plastic KC->MBON contribution (scatter-add)
            pre_spk = arrivals[self.pl_pre]
            nz = pre_spk != 0.0
            if nz.any():
                ginc += np.bincount(
                    self.pl_post[nz], weights=self.w_plastic[nz] * pre_spk[nz], minlength=self.n
                )
            self.g += ginc

        # threshold, reset, refractory
        spikes = (self.v > self.v_th) & active
        if spikes.any():
            self.v[spikes] = self.v_rst
            self.g[spikes] = 0.0
            self.refr[spikes] = self.refr_steps
        self.refr[~active] -= 1

        # record spikes for future delivery
        self._ring[self._head] = spikes
        self._head = (self._head + 1) % self.delay_steps
        return spikes

    def run_counts(self, dur_ms: float) -> np.ndarray:
        steps = int(round(dur_ms / self.dt))
        counts = np.zeros(self.n, dtype=np.int64)
        for _ in range(steps):
            counts += self._step()
        return counts
