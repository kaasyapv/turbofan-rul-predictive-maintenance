import numpy as np
from src.prep import windows


def test_windows_stay_inside_one_engine():
    unit = np.repeat([1, 2], 40)
    x = np.where(unit == 1, 1.0, 2.0)[:, None]
    w = windows(x, unit)
    assert w.shape == (80, 30, 1)
    assert (w[:40] == 1).all() and (w[40:] == 2).all()


def test_short_history_is_front_padded_with_first_row():
    x = np.arange(5.0)[:, None]
    w = windows(x, np.ones(5))
    assert (w[0, :-1] == 0).all() and w[0, -1] == 0
    assert list(w[3, -4:, 0]) == [0, 1, 2, 3] and (w[3, :-4] == 0).all()


def test_last_only_gives_one_window_per_engine():
    unit = np.repeat([1, 2, 3], 35)
    x = np.arange(105.0)[:, None]
    w = windows(x, unit, last_only=True)
    assert w.shape == (3, 30, 1)
    assert list(w[:, -1, 0]) == [34, 69, 104]
