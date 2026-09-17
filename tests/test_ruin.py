"""Gambler's ruin sanity check for the fair game (no brain needed).

For a fair, even-money game, a $1-stake bettor starting at ``start`` heading for
``target`` (ruin at $0) reaches the target with probability ``start / target``.
This validates the README's "~25% rich / ~75% broke on a truly random game".
"""

from flygambler.game import FairBook


def test_fair_gamblers_ruin_probability():
    start, target, stake, N = 25, 100, 1, 2000
    book = FairBook(odds=2.0, seed=0)
    cue = book.new_cue()          # cue is irrelevant on a fair book
    rich = 0
    for _ in range(N):
        bank = start
        while 0 < bank < target:
            bank += stake * book.settle(cue).reward   # +1 win / -1 loss, fair
        if bank >= target:
            rich += 1
    p = rich / N
    assert abs(p - start / target) < 0.035            # theory: 0.25, within sampling error
