import numpy as np
from flygambler.game import FairBook, RiggedBook, LearnableBook, make_book


def test_fair_is_zero_edge():
    b = FairBook(odds=2.0, seed=0)
    assert abs(b.p_win(b.new_cue()) - 0.5) < 1e-9
    assert abs(b.ev(b.new_cue())) < 1e-9
    assert abs(b.house_edge) < 1e-6


def test_rigged_is_negative_ev():
    b = RiggedBook(odds=2.0, edge=0.06, seed=0)
    assert b.ev(b.new_cue()) < 0
    assert b.house_edge > 0.05


def test_learnable_has_good_and_bad_cues():
    b = LearnableBook(odds=2.0, seed=1)
    evs = [b.ev(b.new_cue()) for _ in range(600)]
    assert max(evs) > 0 and min(evs) < 0          # a real edge exists to find
    ps = [b.p_win(b.new_cue()) for _ in range(300)]
    assert all(0.0 <= p <= 1.0 for p in ps)


def test_cue_is_sparse():
    b = make_book("fair", n_channels=16, active_channels=6, seed=2)
    c = b.new_cue()
    assert len(c) == 16 and int((c > 0).sum()) == 6


def test_make_book_aliases():
    assert isinstance(make_book("random"), FairBook)
    assert isinstance(make_book("house"), RiggedBook)
    assert isinstance(make_book("beatable"), LearnableBook)


def test_settle_pays_correctly():
    b = FairBook(odds=2.0, seed=3)
    outs = [b.settle(b.new_cue()) for _ in range(400)]
    for o in outs:
        assert o.reward == (1.0 if o.won else -1.0)   # even odds: +1 win / -1 loss
    wr = np.mean([o.won for o in outs])
    assert 0.4 < wr < 0.6                             # ~fair
