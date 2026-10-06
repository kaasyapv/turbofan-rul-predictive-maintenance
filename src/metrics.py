import numpy as np


def clip(pred):
    return np.clip(pred, 0, 125)


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def score(pred, true) -> float:
    """NASA asymmetric score. Late predictions (d > 0) cost more than early ones."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))


def flag(pred, true, limit=30) -> tuple:
    """Precision and recall of 'pull this engine' against engines truly at or under the limit."""
    hit = (pred <= limit) & (true <= limit)
    return float(hit.sum() / max((pred <= limit).sum(), 1)), float(hit.sum() / max((true <= limit).sum(), 1))


def report(pred, true) -> dict:
    pred = clip(pred)
    p, r = flag(pred, true)
    return {"rmse": rmse(pred, true), "score": score(pred, true), "precision": p, "recall": r}
