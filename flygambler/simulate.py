"""Run a fly's gambling life to ruin or riches, and run whole populations.

A "life" is a loop of rounds. Each round the fly sees a cue, bets or passes,
and (if it bet) gets a reward that fires dopamine/pain and rewrites its
KC->MBON weights. The life ends when the fly:

* **goes broke** -- bankroll can no longer cover a stake ("kills itself", the
  classic gambler's ruin), or
* **gets rich** -- bankroll reaches the target, or
* **times out** -- reaches ``max_rounds``.

Every round is logged so the run can be plotted and animated in the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .brain import FlyBrain
from .game import Sportsbook


RUIN, RICH, TIMEOUT = "ruin", "rich", "timeout"


@dataclass
class SimConfig:
    bankroll0: float = 25.0
    stake: float = 1.0
    target: float = 100.0            # "rich" threshold
    max_rounds: int | None = 2000    # None => run a life with no round cap


@dataclass
class RunLog:
    meta: dict
    rounds: list = field(default_factory=list)

    @property
    def outcome(self) -> str:
        return self.meta["outcome"]

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": self.meta, "rounds": self.rounds}))
        return path

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rounds)


def _f(x) -> float:
    return float(x)


def run_life(
    brain: FlyBrain,
    book: Sportsbook,
    cfg: SimConfig | None = None,
    *,
    on_round=None,
    calibrate: bool = True,
    store_rounds: bool = True,
    verbose: bool = False,
) -> RunLog:
    """Simulate one fly until ruin, riches, or timeout.

    ``cfg.max_rounds=None`` runs until ruin/riches with no cap. ``store_rounds``
    False keeps only running summaries (for endless play -> bounded memory).
    """
    from collections import deque
    cfg = cfg or SimConfig()
    if calibrate:
        # centre the bet decision on this book's actual cue distribution
        brain.calibrate([book.new_cue() for _ in range(40)])
    bankroll = cfg.bankroll0
    peak = bankroll
    rounds: list[dict] = []
    outcome = TIMEOUT
    tail = deque(maxlen=100)          # (is_bet, mood) for gave-up detection
    n_bets = n_wins = n_losses = 0
    cum_pain = cum_hurt = 0.0
    final_mood = 0.0
    t = 0

    limit = cfg.max_rounds
    while limit is None or t < limit:
        t += 1
        if bankroll < cfg.stake:            # cannot cover a bet -> ruined
            outcome = RUIN
            break

        cue = book.new_cue()
        cue_on = np.nonzero(np.asarray(cue) > 0)[0].tolist()
        dec, ctx = brain.decide(cue)
        did_bet = dec.action == "bet"
        ev = book.ev(cue)
        p_win = book.p_win(cue)

        won = None
        money_delta = 0.0
        reward = 0.0
        if did_bet:
            out = book.settle(cue)
            reward = out.reward
            money_delta = cfg.stake * reward
            bankroll += money_delta
            won = out.won

        fb = brain.receive_outcome(ctx, reward, did_bet)
        peak = max(peak, bankroll)

        if did_bet:
            n_bets += 1
            n_wins += 1 if won else 0
            n_losses += 0 if won else 1
        cum_pain += fb.pain
        cum_hurt += max(-fb.rpe, 0.0)
        final_mood = fb.mood
        tail.append((did_bet, fb.mood))

        rec = {
            "t": t,
            "bankroll": _f(bankroll),
            "action": dec.action,
            "cue": cue_on,
            "p_bet": _f(dec.p_bet),
            "drive": _f(dec.drive),
            "kc_frac": _f(dec.kc_frac),
            "kc_active": int(dec.kc_active),
            "mbon_spikes": int(dec.mbon_spikes),
            "won": (None if won is None else bool(won)),
            "money_delta": _f(money_delta),
            "reward": _f(reward),
            "rpe": _f(fb.rpe),
            "dopamine": _f(fb.dopamine),
            "pain": _f(fb.pain),
            "hedonic": _f(fb.hedonic),
            "mood": _f(fb.mood),
            "value": _f(fb.value),
            "cue_ev": _f(ev),
            "cue_pwin": _f(p_win),
            "dw_mean": _f(fb.dw_mean),
        }
        if store_rounds:
            rounds.append(rec)
        if on_round is not None:
            on_round(rec)

        if bankroll >= cfg.target:
            outcome = RICH
            break

    n_rounds = max(0, t - (1 if outcome == RUIN else 0))
    # "gave up": not ruined, but disengaged (barely betting) over the last stretch while sad
    tail_bet_rate = (sum(1 for b, _ in tail if b) / len(tail)) if tail else 0.0
    gave_up = bool(outcome != RUIN and tail_bet_rate < 0.1 and final_mood < -0.5)
    meta = {
        "outcome": outcome,
        "n_rounds": n_rounds,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "cum_pain": _f(cum_pain),
        "cum_hurt": _f(cum_hurt),
        "final_mood": _f(final_mood),
        "gave_up": gave_up,
        "final_bankroll": _f(bankroll),
        "peak_bankroll": _f(peak),
        "bankroll0": cfg.bankroll0,
        "stake": cfg.stake,
        "target": cfg.target,
        "max_rounds": cfg.max_rounds,
        "book": type(book).__name__,
        "odds": book.odds,
        "n_channels": int(getattr(book, "n_channels", 16)),
        "active_channels": int(getattr(book, "active_channels", 6)),
        "n_neurons": int(brain.engine.n_neurons),
        "n_kc": int(len(brain.kc)),
        "n_mbon": int(len(brain.mbon)),
        "scope": brain.cfg.side,
    }
    if verbose:
        print(
            f"    life: {outcome.upper()} after {n_rounds} rounds "
            f"({n_bets} bets, {n_wins} wins) | final ${bankroll:.1f} peak ${peak:.1f}"
        )
    return RunLog(meta=meta, rounds=rounds)


@dataclass
class PopulationResult:
    summaries: list = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.summaries)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summaries))
        return path


def run_population(
    brain: FlyBrain,
    book_factory,
    n_flies: int = 50,
    cfg: SimConfig | None = None,
    *,
    base_seed: int = 1000,
    verbose: bool = True,
) -> PopulationResult:
    """Run many flies (reusing one built network) and collect their fates."""
    cfg = cfg or SimConfig()
    # calibrate once on this book family (same cue distribution + fresh weights for all flies)
    brain.reset(seed=base_seed)
    brain.calibrate([book_factory(base_seed).new_cue() for _ in range(40)])
    summaries = []
    for k in range(n_flies):
        brain.reset(seed=base_seed + k)
        book = book_factory(base_seed + k)
        log = run_life(brain, book, cfg, calibrate=False)
        s = dict(log.meta)
        s["fly"] = k
        summaries.append(s)
        if verbose and (k + 1) % max(1, n_flies // 10) == 0:
            done = k + 1
            ruined = sum(1 for x in summaries if x["outcome"] == RUIN)
            rich = sum(1 for x in summaries if x["outcome"] == RICH)
            print(f"    {done}/{n_flies} flies | ruined {ruined} rich {rich}")
    return PopulationResult(summaries=summaries)


def run_forever(
    brain: FlyBrain,
    book_factory,
    cfg: SimConfig | None = None,
    *,
    base_seed: int = 0,
    on_round=None,
    on_life=None,
    lives_path: str | Path | None = None,
    max_lives: int | None = None,
    verbose: bool = False,
):
    """Play endless lives on one brain: respawn on ruin/riches, forever.

    Each completed life's summary is appended to ``lives_path`` (JSONL) -- the
    permanent record of "what happened" to every fly. Runs until ``max_lives``
    (None = forever, i.e. until interrupted). ``on_round(rec)`` streams every
    round for a live view; ``on_life(meta, counts)`` fires when a fly's life ends.
    """
    import itertools
    cfg = cfg or SimConfig()
    path = Path(lives_path) if lives_path else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    # calibrate once: same cue family and fresh weights for every life
    brain.reset(seed=base_seed)
    brain.calibrate([book_factory(base_seed).new_cue() for _ in range(40)])

    counts = {"lives": 0, RUIN: 0, RICH: 0, TIMEOUT: 0}
    it = itertools.count() if max_lives is None else range(max_lives)
    for k in it:
        seed = base_seed + k
        brain.reset(seed=seed)
        book = book_factory(seed)
        log = run_life(brain, book, cfg, on_round=on_round, calibrate=False, store_rounds=False)
        m = dict(log.meta); m["life"] = k + 1; m["seed"] = seed
        counts["lives"] += 1
        counts[m["outcome"]] = counts.get(m["outcome"], 0) + 1
        if path:
            with path.open("a") as f:
                f.write(json.dumps(m) + "\n")
        if on_life is not None:
            on_life(m, counts)
        if verbose:
            print(f"life {k+1:>4}: {m['outcome'].upper():7s} in {m['n_rounds']:>4} rounds, "
                  f"final ${m['final_bankroll']:>5.0f}  (broke {counts[RUIN]} / rich {counts[RICH]})")
    return counts
