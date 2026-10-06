import numpy as np
from src.metrics import score


def test_late_costs_more_than_early():
    true = np.array([50.0])
    assert score(true + 10, true) > score(true - 10, true)


def test_perfect_is_zero():
    assert score(np.array([30.0]), np.array([30.0])) == 0
