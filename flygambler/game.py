"""Betting environments the fly gambles in.

Each round the book shows a **cue** (a feature vector -- think a matchup's
stats). The fly bets one unit at fixed decimal odds ``O`` or passes.
Winning a 1-unit bet returns a profit of ``O - 1``; losing costs 1 unit.

Two books, to contrast how a real reward-learning circuit behaves:

* ``RiggedBook`` -- the outcome is pure chance with a built-in house edge
  (the "vig"), and the cue predicts nothing. Every bet is negative expected
  value; the only winning move is *not to play*. Intermittent wins (a
  variable-ratio schedule) are exactly what drives compulsive gambling, so
  this tests whether the fly chases losses or learns to quit -- and whether
  homeostatic forgetting makes it relapse.

* ``LearnableBook`` -- the cue genuinely predicts the win probability (a hidden
  function of the features). Good cues are positive expected value, bad cues
  negative. A fly that learns to bet good cues and pass bad ones can actually
  get rich. This tests whether the mushroom-body circuit can find a real edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Outcome:
    won: bool
    odds: float
    reward: float   # profit per unit stake: +(odds-1) on a win, -1 on a loss
    p_win: float    # true win probability for this cue
    ev: float       # expected value per unit stake at these odds (odds*p_win - 1)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Sportsbook:
    """Base class: fixed decimal odds, cue generation, settlement.

    Cues are **sparse**: exactly ``active_channels`` of ``n_channels`` feature
    channels are "on" (value 1), the rest 0. Sparse cues produce distinct
    Kenyon-cell codes, which is what lets the mushroom body assign credit
    cleanly and actually learn which cues are worth betting.
    """

    def __init__(self, n_channels: int = 16, odds: float = 2.0,
                 active_channels: int = 6, seed: int = 0):
        self.n_channels = n_channels
        self.odds = float(odds)
        self.active_channels = int(active_channels)
        self.rng = np.random.default_rng(seed)

    def new_cue(self) -> np.ndarray:
        c = np.zeros(self.n_channels)
        on = self.rng.choice(self.n_channels, self.active_channels, replace=False)
        c[on] = 1.0
        return c

    def p_win(self, cue: np.ndarray) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def ev(self, cue: np.ndarray) -> float:
        return self.odds * self.p_win(cue) - 1.0

    def settle(self, cue: np.ndarray) -> Outcome:
        p = self.p_win(cue)
        won = bool(self.rng.random() < p)
        reward = (self.odds - 1.0) if won else -1.0
        return Outcome(won=won, odds=self.odds, reward=reward, p_win=p, ev=self.odds * p - 1.0)

    @property
    def house_edge(self) -> float:
        """Average house edge over random cues (negative EV to the bettor)."""
        cues = self.rng.random((512, self.n_channels))
        return -float(np.mean([self.ev(c) for c in cues]))


class RiggedBook(Sportsbook):
    """Pure chance with a house edge; the cue is irrelevant. Optimal: never bet."""

    def __init__(self, n_channels: int = 16, odds: float = 2.0, edge: float = 0.06,
                 active_channels: int = 6, seed: int = 0):
        super().__init__(n_channels, odds, active_channels, seed)
        # break-even win prob is 1/odds; shave off `edge` so every bet is -EV
        self.base_p = (1.0 / self.odds) - edge
        self.edge = edge

    def p_win(self, cue: np.ndarray) -> float:
        return self.base_p


class LearnableBook(Sportsbook):
    """The cue predicts the outcome; good cues are +EV, bad cues -EV.

    Win probability is a hidden linear function of which channels are active,
    centred so that a random cue is roughly a coin-flip, then squashed into
    ``p_range``. A fly that learns ``theta`` (implicitly, in its KC->MBON
    weights) bets the good cues and passes the bad ones.
    """

    def __init__(
        self,
        n_channels: int = 16,
        odds: float = 2.0,
        active_channels: int = 6,
        slope: float = 4.0,
        p_range: tuple[float, float] = (0.12, 0.88),
        seed: int = 0,
    ):
        super().__init__(n_channels, odds, active_channels, seed)
        self.theta = self.rng.normal(size=n_channels)
        self.theta /= np.linalg.norm(self.theta)
        self.slope = slope
        self.p_lo, self.p_hi = p_range
        # centre so the average sparse cue is ~a coin flip
        self.bias = float(self.theta.sum()) * self.active_channels / self.n_channels

    def p_win(self, cue: np.ndarray) -> float:
        s = float(self.theta @ np.asarray(cue)) - self.bias
        p = _sigmoid(self.slope * s)
        return self.p_lo + (self.p_hi - self.p_lo) * p


def make_book(kind: str, **kwargs) -> Sportsbook:
    kind = kind.lower()
    if kind in ("rigged", "house", "casino"):
        return RiggedBook(**kwargs)
    if kind in ("learnable", "beatable", "edge"):
        return LearnableBook(**kwargs)
    raise ValueError(f"unknown book kind: {kind!r}")
