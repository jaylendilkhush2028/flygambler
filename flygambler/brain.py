"""FlyBrain: the real mushroom-body circuit turned into a gambler that learns.

A round of "thinking and acting":

1. **Encode the cue.** A betting cue (a feature vector -- think a matchup's
   stats/odds) is mapped to drive on a set of real antennal-lobe projection
   neurons (ALPNs). The real ALPN->KC connectome wiring expands this into a
   sparse Kenyon-cell code (the fly's high-dimensional representation).
2. **Think.** The subnetwork spikes for ~100 ms. MBONs integrate KC input
   through the *current* (learned) KC->MBON weights.
3. **Act.** Total MBON drive -> a bet-drive D; P(bet) = sigmoid((D - D_ref)/T).
   The fly bets or passes (with a little exploration).
4. **Dopamine / pain.** After a bet, reward-prediction error delta = reward -
   value. delta>0 fires the **PAM** reward-dopamine cluster; delta<0 fires the
   **PPL1** punishment cluster ("pain"). These real neurons actually spike, and
   their firing is the dopamine/pain signal we report.
5. **Learn.** A dopamine-gated three-factor rule rewrites the KC->MBON weights
   of the *most responsive* Kenyon cells (a k-winner-take-all set -- the APL's
   sparsening role): potentiate on reward, depress on punishment, with slow
   homeostatic decay back toward the anatomical baseline (forgetting / relapse).

Honest note: steps 1-3 run on the *real connectome* (real neurons, real
synapses). The plasticity in step 5 is our biologically-motivated addition --
the published whole-brain model has fixed weights and does not learn. The rule
follows the biology of dopamine-driven KC->MBON plasticity (Hige, Cohn, Owald,
Aso et al.), collapsed to a single "approach/bet" readout for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import ConnectomeTables, Circuit, extract_circuit, mushroom_body_ids
from .fastlif import FastLIF


@dataclass
class BrainConfig:
    side: str = "right"                 # hemisphere for the mushroom-body subnetwork
    # cue encoding
    n_channels: int = 16                # feature channels mapped onto ALPN groups
    cue_active_channels: int = 6        # active channels per cue (match the book's sparsity)
    cue_drive_mV: float = 55.0          # ALPN drive at feature value 1.0
    t_cue_ms: float = 100.0             # cue presentation window
    # decision
    temperature: float = 0.7            # softness of the bet decision
    explore_eps: float = 0.05           # exploration floor on P(bet)
    adapt_ref: float = 0.01             # slow drift of the decision reference
    bet_bias: float = 0.0               # >0 => more cautious (self-control intervention)
    # dopamine / teaching
    simulate_dopamine: bool = True      # actually fire PAM/PPL1 so dopamine/pain are real spikes
    t_teach_ms: float = 40.0            # dopamine delivery window
    dan_drive_mV: float = 45.0          # DAN drive at |rpe| = 1
    value_lr: float = 0.05              # EMA rate for the reward baseline (critic)
    # plasticity (KC->MBON)
    learn_rate: float = 3.0             # weight step per unit (rpe * eligibility), in mV
    reward_scale: float = 1.0           # gain on learning from WINS  (0 => wins don't reinforce)
    punish_scale: float = 1.0           # gain on learning from LOSSES (0 => losses don't teach)
    kc_topk_frac: float = 0.08          # only the top-k% most active KCs learn (APL sparsening)
    kc_ref: float = 3.0                 # KC spike count that saturates eligibility
    decay: float = 0.004                # homeostatic forgetting toward baseline (higher => extinction)
    w_max_mult: float = 4.0             # cap plastic weights at this * max(anatomical weight)
    # hedonic trace (fast pleasure/pain, visualisation only)
    hedonic_gain: float = 1.0
    hedonic_decay: float = 0.6
    # mood / despair — a SLOW internal state that builds with sustained loss and,
    # when `withdrawal` > 0, suppresses betting (learned-helplessness / "too sad
    # to keep playing"). Defaults leave behaviour unchanged (withdrawal = 0).
    mood_gain: float = 0.15             # how strongly each outcome moves mood
    mood_recovery: float = 0.99         # per-round drift of mood back toward neutral (slow => persistent)
    win_mood_factor: float = 0.9        # wins lift mood nearly as much as losses lower it (mild loss-aversion)
    withdrawal: float = 0.0             # how strongly low mood suppresses betting (0 => off)
    # engine
    dt_ms: float = 0.5
    seed: int = 0


@dataclass
class Decision:
    action: str          # "bet" or "pass"
    p_bet: float
    drive: float         # bet-drive D this round
    kc_active: int
    kc_frac: float
    mbon_spikes: int
    alpn_spikes: int


@dataclass
class Feedback:
    rpe: float
    dopamine: float      # PAM reward-cluster spikes (or |rpe| proxy)
    pain: float          # PPL1 punishment-cluster spikes (or |rpe| proxy)
    hedonic: float
    mood: float          # slow despair/mood state (negative = sad)
    value: float
    dw_mean: float


class FlyBrain:
    def __init__(
        self,
        tables: ConnectomeTables | None,
        config: BrainConfig | None = None,
        *,
        engine: FastLIF | None = None,
        include_ids: np.ndarray | None = None,
        verbose: bool = True,
    ):
        self.cfg = config or BrainConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        if engine is None:
            if include_ids is None:
                include_ids = mushroom_body_ids(tables, side=self.cfg.side)
            circuit = extract_circuit(tables, include_ids, verbose=verbose)
            engine = FastLIF(circuit, dt_ms=self.cfg.dt_ms)
        self.engine = engine

        self.alpn = engine.role_idx["ALPN"]
        self.kc = engine.role_idx["KC"]
        self.mbon = engine.role_idx["MBON"]
        self.pam = engine.role_idx["PAM"]
        self.ppl1 = engine.role_idx["PPL1"]

        # Opponent MBON channels: approach ("bet") vs avoidance ("pass").
        # Real MBONs split into compartments of opposite valence that dopamine
        # pushes in opposite directions; the decision reads their difference,
        # which cancels global drift while preserving per-cue learning.
        split_rng = np.random.default_rng(self.cfg.seed + 9973)
        order = split_rng.permutation(len(self.mbon))
        plus_local = set(self.mbon[order[: len(self.mbon) // 2]].tolist())  # approach pool
        self.mbon_is_plus = np.array([int(m) in plus_local for m in self.mbon], dtype=bool)

        # partition ALPNs into feature channels
        perm = self.rng.permutation(self.alpn)
        self.channels = np.array_split(perm, self.cfg.n_channels)

        # plastic-weight working copy (mV) + anatomical baseline
        self.w = engine.get_plastic().astype(np.float64).copy()
        self.w0 = engine.w0_plastic.copy()
        self.w_cap = self.cfg.w_max_mult * float(self.w0.max()) if self.w0.size else 1.0
        self.pre_kc = engine.plastic_pre_kc
        self.post_mbon = engine.plastic_post_mbon
        # +1 if this KC->MBON synapse targets the approach pool, -1 if avoidance
        self.plastic_sign = np.where(np.isin(self.post_mbon, self.mbon[self.mbon_is_plus]), 1.0, -1.0)

        self.value = 0.0
        self.hedonic = 0.0
        self.mood = 0.0          # slow despair state (negative = sad); see BrainConfig
        self.n_rounds = 0

        self.D_ref0, self.D_scale0 = self._calibrate(n=40, verbose=verbose)
        self.D_ref, self.D_scale = self.D_ref0, self.D_scale0

    def _random_cue(self) -> np.ndarray:
        """A sparse cue matching the book's structure (for calibration)."""
        c = np.zeros(self.cfg.n_channels)
        on = self.rng.choice(self.cfg.n_channels, self.cfg.cue_active_channels, replace=False)
        c[on] = 1.0
        return c

    def calibrate(self, cues) -> None:
        """Set the naive bet-drive reference from a sample of representative cues."""
        drives = np.array([self._run_cue(c)[4] for c in cues])
        self.D_ref0 = float(drives.mean())
        self.D_scale0 = float(drives.std() + 1e-6)
        self.D_ref, self.D_scale = self.D_ref0, self.D_scale0

    def reset(self, seed: int | None = None) -> None:
        """Restore naive state (fresh weights, critic, hedonic) for a new fly."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.w = self.w0.copy()
        self.engine.set_plastic(self.w)
        self.value = 0.0
        self.hedonic = 0.0
        self.mood = 0.0
        self.n_rounds = 0
        self.D_ref, self.D_scale = self.D_ref0, self.D_scale0

    # ----------------------------------------------------------------- helpers
    def encode(self, cue: np.ndarray) -> None:
        self.engine.reset_state()
        drive = np.zeros(self.engine.n_neurons)
        for c, feat in enumerate(cue):
            drive[self.channels[c]] = float(feat) * self.cfg.cue_drive_mV
        self.engine.set_stim_vector(drive)

    def _run_cue(self, cue: np.ndarray):
        self.encode(cue)
        delta = self.engine.run_counts(self.cfg.t_cue_ms)
        mbon_c = delta[self.mbon]
        # bet-drive = approach-pool activity minus avoidance-pool activity
        drive = float(mbon_c[self.mbon_is_plus].sum() - mbon_c[~self.mbon_is_plus].sum())
        return delta, delta[self.kc], mbon_c, delta[self.alpn], drive

    def _calibrate(self, n: int, verbose: bool):
        drives = []
        for _ in range(n):
            _, _, _, _, D = self._run_cue(self._random_cue())
            drives.append(D)
        drives = np.array(drives)
        ref, scale = float(drives.mean()), float(drives.std() + 1e-6)
        if verbose:
            print(f"    calibrated bet-drive: D_ref={ref:.1f}  D_scale={scale:.1f}")
        return ref, scale

    def _deliver_dopamine(self, valence: str, magnitude: float) -> tuple[float, float]:
        if not self.cfg.simulate_dopamine:
            return (magnitude, 0.0) if valence == "reward" else (0.0, magnitude)
        self.engine.reset_state()
        target = self.pam if valence == "reward" else self.ppl1
        drive = np.zeros(self.engine.n_neurons)
        drive[target] = min(magnitude, 1.5) * self.cfg.dan_drive_mV
        self.engine.set_stim_vector(drive)
        d = self.engine.run_counts(self.cfg.t_teach_ms)
        return float(d[self.pam].sum()), float(d[self.ppl1].sum())

    # ---------------------------------------------------------------- main API
    def decide(self, cue: np.ndarray) -> tuple[Decision, dict]:
        delta, kc_c, mbon_c, alpn_c, drive = self._run_cue(cue)
        z = (drive - self.D_ref) / (self.cfg.temperature * self.D_scale) - self.cfg.bet_bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        eps = self.cfg.explore_eps
        p_bet = eps + (1.0 - 2.0 * eps) * p
        # Sustained sadness gates betting toward the exploration floor -- applied
        # multiplicatively on P(bet) so it works regardless of the drive's scale
        # (a raw z-penalty gets swamped once learning drifts the drive).
        if self.cfg.withdrawal > 0:
            sad = max(0.0, -self.mood)
            gate = 1.0 / (1.0 + np.exp(np.clip(self.cfg.withdrawal * (sad - 2.0), -30.0, 30.0)))
            p_bet = eps + (p_bet - eps) * gate
        if self.cfg.adapt_ref > 0:
            self.D_ref += self.cfg.adapt_ref * (drive - self.D_ref)
        action = "bet" if self.rng.random() < p_bet else "pass"
        dec = Decision(
            action=action, p_bet=float(p_bet), drive=drive,
            kc_active=int((kc_c > 0).sum()), kc_frac=float((kc_c > 0).mean()),
            mbon_spikes=int(mbon_c.sum()), alpn_spikes=int(alpn_c.sum()),
        )
        return dec, {"kc_counts": kc_c}

    def receive_outcome(self, ctx: dict, reward: float, did_bet: bool) -> Feedback:
        self.n_rounds += 1
        self.mood *= self.cfg.mood_recovery      # slow drift back toward neutral every round
        if not did_bet:
            self.hedonic *= self.cfg.hedonic_decay
            return Feedback(0.0, 0.0, 0.0, self.hedonic, self.mood, self.value, 0.0)

        rpe = reward - self.value
        self.value += self.cfg.value_lr * rpe
        # mood follows ACTUAL fortune: up when it wins money, down when it loses
        # (uses the round reward, not prediction error, so a winning fly is happy)
        self.mood += self.cfg.mood_gain * (reward if reward < 0 else self.cfg.win_mood_factor * reward)
        self.mood = float(np.clip(self.mood, -4.0, 3.0))
        valence = "reward" if rpe >= 0 else "punish"
        pam_spk, ppl1_spk = self._deliver_dopamine(valence, abs(rpe))

        # eligibility: k-winner-take-all over Kenyon cells (only most-responsive learn)
        kc_counts = ctx["kc_counts"].astype(np.float64)
        n_kc = len(kc_counts)
        k = max(1, int(round(self.cfg.kc_topk_frac * n_kc)))
        elig_kc = np.zeros(n_kc)
        if k < n_kc:
            top = np.argpartition(kc_counts, -k)[-k:]
            sel = top[kc_counts[top] > 0]
        else:
            sel = np.nonzero(kc_counts > 0)[0]
        elig_kc[sel] = np.clip(kc_counts[sel] / self.cfg.kc_ref, 0.0, 1.0)

        elig_full = np.zeros(self.engine.n_neurons)
        elig_full[self.kc] = elig_kc
        elig = elig_full[self.pre_kc]

        # dopamine-gated three-factor update on KC->MBON weights.
        # Sign flips by target pool: reward strengthens approach & weakens
        # avoidance (and vice-versa for punishment) -- the opponent rule.
        # reward_scale / punish_scale let interventions block one side of learning.
        gain = self.cfg.reward_scale if rpe >= 0 else self.cfg.punish_scale
        dw = gain * self.cfg.learn_rate * rpe * elig * self.plastic_sign
        self.w += dw
        self.w += self.cfg.decay * (self.w0 - self.w)   # homeostatic forgetting
        np.clip(self.w, 0.0, self.w_cap, out=self.w)
        self.engine.set_plastic(self.w)

        self.hedonic = self.cfg.hedonic_decay * self.hedonic + self.cfg.hedonic_gain * rpe
        return Feedback(
            rpe=float(rpe), dopamine=float(pam_spk), pain=float(ppl1_spk),
            hedonic=float(self.hedonic), mood=float(self.mood), value=float(self.value),
            dw_mean=float(np.abs(dw[elig > 0]).mean()) if np.any(elig > 0) else 0.0,
        )


def make_brain(
    tables: ConnectomeTables,
    config: BrainConfig | None = None,
    *,
    verbose: bool = True,
) -> FlyBrain:
    """Build a FlyBrain on the fast engine over the mushroom-body subnetwork."""
    config = config or BrainConfig()
    ids = mushroom_body_ids(tables, side=config.side)
    circuit = extract_circuit(tables, ids, verbose=verbose)
    engine = FastLIF(circuit, dt_ms=config.dt_ms)
    return FlyBrain(None, config, engine=engine, verbose=verbose)
