"""Matplotlib figures for a fly's gambling life and for a population."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# outcome colours
_OC = {"ruin": "#d1495b", "rich": "#2a9d8f", "timeout": "#8d99ae"}


def _roll(x, w=25):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x
    w = max(1, min(w, len(x)))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def plot_life(log, path: str | Path, title: str | None = None):
    """Four-panel life story: bankroll, dopamine/pain, engagement, learning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = log.to_dataframe()
    if df.empty:
        return None
    t = df["t"].to_numpy()
    outcome = log.meta["outcome"]

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        title or f"{log.meta['book']} · {outcome.upper()} after {log.meta['n_rounds']} rounds "
        f"(final ${log.meta['final_bankroll']:.0f})",
        fontsize=14, fontweight="bold",
    )

    # (1) bankroll
    ax[0].plot(t, df["bankroll"], color="#264653", lw=1.6)
    ax[0].axhline(0, color=_OC["ruin"], ls="--", lw=1, alpha=.7)
    ax[0].axhline(log.meta["target"], color=_OC["rich"], ls="--", lw=1, alpha=.7)
    ax[0].axhline(log.meta["bankroll0"], color="#999", ls=":", lw=1, alpha=.6)
    ax[0].fill_between(t, 0, df["bankroll"], color=_OC.get(outcome, "#264653"), alpha=.08)
    ax[0].set_ylabel("bankroll ($)")
    ax[0].set_title("Money — until ruin ($0, 'kills itself') or riches (target)", loc="left", fontsize=10)

    # (2) dopamine / pain
    wins = df["won"] == True
    losses = df["won"] == False
    ax[1].plot(t, _roll(df["dopamine"]), color="#2a9d8f", lw=1.4, label="dopamine (PAM, reward)")
    ax[1].plot(t, _roll(df["pain"]), color="#d1495b", lw=1.4, label="pain (PPL1, punishment)")
    ax[1].scatter(t[wins], np.zeros(wins.sum()), s=6, color="#2a9d8f", alpha=.5, marker="^")
    ax[1].scatter(t[losses], np.zeros(losses.sum()), s=6, color="#d1495b", alpha=.5, marker="v")
    ax[1].set_ylabel("DAN spikes (smoothed)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("Dopamine on wins, pain on losses (real PAM / PPL1 neurons firing)", loc="left", fontsize=10)

    # (3) engagement
    ax[2].plot(t, _roll((df["action"] == "bet").to_numpy().astype(float)), color="#e76f51", lw=1.6)
    ax[2].plot(t, df["p_bet"], color="#f4a261", lw=.6, alpha=.4)
    ax[2].axhline(0.5, color="#999", ls=":", lw=1)
    ax[2].set_ylim(-.02, 1.02)
    ax[2].set_ylabel("P(bet) / bet rate")
    ax[2].set_title("Engagement — does it keep gambling, quit, or relapse?", loc="left", fontsize=10)

    # (4) learning: on a beatable book, does betting separate good vs bad cues?
    has_both = (df["cue_ev"] > 0).any() and (df["cue_ev"] <= 0).any()
    if has_both:
        pos = df["cue_ev"] > 0
        for mask, col, lab in [(pos, "#2a9d8f", "+EV (good) cues"), (~pos, "#d1495b", "-EV (bad) cues")]:
            sub = df[mask]
            rate = (sub["action"] == "bet").astype(float).rolling(50, min_periods=5).mean()
            ax[3].plot(sub["t"], rate, color=col, lw=1.7, label=f"bet rate on {lab}")
        ax[3].axhline(0.5, color="#999", ls=":", lw=1)
        ax[3].set_ylim(-.02, 1.02)
        ax[3].set_ylabel("bet rate by cue value")
        ax[3].legend(loc="lower left", fontsize=8)
        ax[3].set_title("Learning — it bets good cues more and bad cues less over time", loc="left", fontsize=10)
        axh = ax[3].twinx()
        axh.plot(t, _roll(df["hedonic"]), color="#6a4c93", lw=.9, alpha=.5)
        axh.set_ylabel("hedonic", color="#6a4c93")
    else:
        ax[3].plot(t, _roll(df["hedonic"]), color="#6a4c93", lw=1.5, label="hedonic state (pleasure/pain)")
        ax[3].axhline(0, color="#999", ls=":", lw=1)
        ax[3].set_ylabel("hedonic")
        ax[3].legend(loc="upper left", fontsize=8)
        ax[3].set_title("Hedonic trace — no learnable edge here (rigged book)", loc="left", fontsize=10)
    ax[3].set_xlabel("round")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_temperaments(logs: dict, path: str | Path, title: str | None = None):
    """Overlay several temperaments (name -> RunLog): bankroll, mood, engagement.

    Shows the contrast between chasing losses to ruin and getting too sad to keep
    playing (withdrawal).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#e76f51", "#4a7ab5", "#2a9d8f", "#6a4c93"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(title or "Temperaments on the same rigged book — chase vs. withdraw",
                 fontweight="bold", fontsize=13)
    for i, (name, log) in enumerate(logs.items()):
        df = log.to_dataframe()
        if df.empty:
            continue
        t = df["t"]
        c = palette[i % len(palette)]
        tag = " (gave up)" if log.meta.get("gave_up") else (" (ruined)" if log.meta.get("outcome") == "ruin" else "")
        ax[0].plot(t, df["bankroll"], color=c, lw=1.6, label=name + tag)
        ax[1].plot(t, _roll(df["mood"]), color=c, lw=1.6, label=name)
        ax[2].plot(t, _roll((df["action"] == "bet").to_numpy().astype(float)), color=c, lw=1.6, label=name)
    ax[0].axhline(0, color="#d1495b", ls="--", lw=1)
    ax[0].set_ylabel("bankroll ($)"); ax[0].set_title("Money", loc="left", fontsize=10)
    ax[0].legend(fontsize=9, loc="upper right")
    ax[1].axhline(0, color="#999", ls=":", lw=1)
    ax[1].set_ylabel("mood  (− = sad)")
    ax[1].set_title("Mood / despair — builds as it loses", loc="left", fontsize=10)
    ax[2].set_ylabel("bet rate"); ax[2].set_ylim(-.02, 1.02); ax[2].set_xlabel("round")
    ax[2].set_title("Engagement — does it keep playing, or get too sad and stop?", loc="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_interventions(results, path: str | Path, title: str | None = None):
    """Compare interventions: what reduces ruin and accumulated pain?

    ``results`` is a list of dicts with keys: name, ruin_rate, mean_pain,
    median_survival, mean_final.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(title or "Interventions on a rigged book — what reduces harm?",
                 fontsize=14, fontweight="bold")

    def bars(a, vals, color, ylabel, fmt="{:.0f}"):
        b = a.bar(x, vals, color=color, alpha=.9)
        a.set_xticks(x); a.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        a.set_ylabel(ylabel)
        for xi, v in zip(x, vals):
            a.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
        return b

    # highlight the baseline vs the rest
    cols = ["#8d99ae"] + ["#2a9d8f"] * (len(names) - 1)
    bars(ax[0], [100 * r["ruin_rate"] for r in results], cols, "% of flies ruined", "{:.0f}%")
    ax[0].set_ylim(0, 105)
    ax[0].set_title("Ruin rate (lower = safer)", loc="left", fontsize=10)

    bars(ax[1], [r["mean_pain"] for r in results], cols, "mean accumulated pain (Σ prediction-error)")
    ax[1].set_title("Total pain endured (lower = kinder)", loc="left", fontsize=10)

    bars(ax[2], [r["median_survival"] for r in results], cols, "median rounds survived")
    ax[2].set_title("Survival (higher = longer before ruin)", loc="left", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_population(pop, path: str | Path, title: str | None = None):
    """Outcome split, final-bankroll distribution, and time-to-ruin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pop.to_dataframe()
    if df.empty:
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title or "Population of flies — fates", fontsize=14, fontweight="bold")

    order = ["ruin", "timeout", "rich"]
    counts = [int((df["outcome"] == o).sum()) for o in order]
    ax[0].bar(order, counts, color=[_OC[o] for o in order])
    for i, c in enumerate(counts):
        ax[0].text(i, c, str(c), ha="center", va="bottom", fontsize=10)
    ax[0].set_ylabel("number of flies")
    ax[0].set_title(f"Outcomes (n={len(df)})", loc="left", fontsize=10)

    ax[1].hist(df["final_bankroll"], bins=20, color="#264653", alpha=.85)
    ax[1].axvline(df["bankroll0"].iloc[0], color="#999", ls=":", lw=1, label="start")
    ax[1].set_xlabel("final bankroll ($)")
    ax[1].set_ylabel("flies")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Where they ended up", loc="left", fontsize=10)

    ruined = df[df["outcome"] == "ruin"]
    if len(ruined):
        ax[2].hist(ruined["n_rounds"], bins=20, color=_OC["ruin"], alpha=.85)
        ax[2].set_xlabel("rounds until ruin")
        ax[2].set_ylabel("flies")
    else:
        ax[2].text(.5, .5, "no ruined flies", ha="center", va="center")
    ax[2].set_title("Time to ruin", loc="left", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
