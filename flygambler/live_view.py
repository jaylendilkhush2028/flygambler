"""A live terminal view of the REAL connectome gambling forever.

Streams ``run_forever`` to a ``rich`` dashboard: current fly's bankroll, mood,
decision, and the running tally of lives lived / broke / rich. Runs until you
press Ctrl+C. Every completed life is also archived to a JSONL file.
"""

from __future__ import annotations

import time
from collections import deque

from .simulate import run_forever, SimConfig


_SPARK = "▁▂▃▄▅▆▇█"


def _spark(vals, lo, hi):
    if not vals:
        return ""
    span = (hi - lo) or 1
    out = []
    for v in vals:
        i = int((v - lo) / span * (len(_SPARK) - 1))
        out.append(_SPARK[max(0, min(len(_SPARK) - 1, i))])
    return "".join(out)


def _bar(frac, width, ch="█"):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return ch * n + "·" * (width - n)


def watch_forever(brain, book_factory, cfg: SimConfig | None = None, *,
                  base_seed: int = 0, lives_path="runs/lives.jsonl",
                  temperament: str = "chaser", book_name: str = "rigged",
                  refresh_hz: float = 12.0):
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Group
    from rich.text import Text
    from rich import box

    cfg = cfg or SimConfig()
    st = {
        "life": 1, "round": 0, "bankroll": cfg.bankroll0, "mood": 0.0,
        "action": "—", "result": "—", "won": None,
        "spark": deque(maxlen=60), "counts": {"lives": 0, "ruin": 0, "rich": 0, "timeout": 0},
        "last_life": None,
    }
    st["spark"].append(cfg.bankroll0)
    moodword = lambda m: "content" if m >= -.3 else "uneasy" if m >= -1 else "low" if m >= -2 else "sad" if m >= -3 else "despairing"

    def render():
        b = st["bankroll"]
        head = Text.assemble(("🪰 FlyGambler  ", "bold"), ("· the real fly brain, gambling forever", "dim"))
        # bankroll
        tbl = Table.grid(padding=(0, 1))
        tbl.add_column(justify="right", style="dim"); tbl.add_column()
        col = "green" if b >= cfg.bankroll0 else "red"
        tbl.add_row("bankroll", Text(f"${b:6.1f}  ", style=f"bold {col}") + Text(_bar(b / cfg.target, 34), style=col)
                    + Text(f"  /${cfg.target:.0f}", style="dim"))
        m = st["mood"]
        mcol = "red" if m < -2 else "magenta"
        tbl.add_row("mood", Text(f"{m:+5.2f}  ", style=f"bold {mcol}") + Text(_bar(max(0, -m) / 4, 34), style=mcol)
                    + Text(f"  {moodword(m)}", style=mcol))
        act = st["action"]
        acol = "green" if act == "bet" else "dim"
        rescol = "green" if st["won"] is True else "red" if st["won"] is False else "dim"
        tbl.add_row("decision", Text(act.upper(), style=f"bold {acol}") + Text("   " + st["result"], style=rescol))
        tbl.add_row("trend", Text(_spark(st["spark"], 0, max(cfg.target, max(st["spark"]))), style="yellow"))

        c = st["counts"]
        tally = Text.assemble(
            (f" life #{st['life']} ", "bold on grey15"), (f"  round {st['round']}", "dim"),
            ("      lives ", "dim"), (f"{c['lives']}", "bold"),
            ("   broke ", "dim"), (f"{c['ruin']}", "bold red"),
            ("   rich ", "dim"), (f"{c['rich']}", "bold yellow"),
        )
        last = Text("", style="dim")
        if st["last_life"]:
            L = st["last_life"]
            oc = "RICH" if L["outcome"] == "rich" else "BROKE" if L["outcome"] == "ruin" else "timeout"
            last = Text(f"last fly: {oc} in {L['n_rounds']} rounds, peak ${L['peak_bankroll']:.0f}", style="dim")
        sub = Text(f"{book_name} book · {temperament} · archiving to {lives_path}  ·  Ctrl+C to stop", style="dim")
        return Panel(Group(tbl, Text(""), tally, last, Text(""), sub), title=head, box=box.ROUNDED, padding=(1, 2))

    last_draw = [0.0]
    live_ref = {}

    def on_round(rec):
        st["round"] = rec["t"]; st["bankroll"] = rec["bankroll"]; st["mood"] = rec["mood"]
        st["action"] = rec["action"]; st["won"] = rec["won"]
        d = rec["money_delta"]
        st["result"] = ("WON +$%.0f" % abs(d)) if rec["won"] is True else ("LOST -$%.0f" % abs(d)) if rec["won"] is False else "(passed)"
        st["spark"].append(rec["bankroll"])
        now = time.time()
        if now - last_draw[0] >= 1.0 / refresh_hz:
            last_draw[0] = now
            live_ref["live"].update(render())

    def on_life(meta, counts):
        st["last_life"] = meta; st["counts"] = counts
        st["life"] = counts["lives"] + 1
        live_ref["live"].update(render())

    with Live(render(), refresh_per_second=refresh_hz, screen=False) as live:
        live_ref["live"] = live
        try:
            run_forever(brain, book_factory, cfg, base_seed=base_seed,
                        on_round=on_round, on_life=on_life, lives_path=lives_path)
        except KeyboardInterrupt:
            pass
    c = st["counts"]
    print(f"\nStopped. Lives lived: {c['lives']}  (broke {c['ruin']}, rich {c['rich']}).  Archive: {lives_path}")
