"""Numpy only preprocessing, shared by training and the API."""
import json
import numpy as np

WINDOW = 30


def nearest(centers, settings):
    d = ((settings[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1)
    return d.argmin(1)


def transform(prep: dict, settings, sensors):
    """Standardise each row with the stats of its operating condition."""
    label = nearest(prep["centers"], np.asarray(settings, float))
    x = np.asarray(sensors, float)[:, prep["keep"]]
    return (x - np.asarray(prep["means"])[label]) / np.asarray(prep["stds"])[label]


def windows(x, unit, last_only=False, size=WINDOW):
    """Last `size` rows of the same engine for every row, or only each engine's final row."""
    unit = np.asarray(unit)
    out = []
    for u in dict.fromkeys(unit.tolist()):
        e = x[unit == u]
        # early cycles have no history yet, so repeat the first row
        e = np.concatenate([np.repeat(e[:1], size - 1, 0), e])
        w = np.lib.stride_tricks.sliding_window_view(e, size, axis=0).transpose(0, 2, 1)
        out.append(w[-1:] if last_only else w)
    return np.concatenate(out)


def save(prep: dict, path: str):
    with open(path, "w") as f:
        json.dump(prep, f)


def read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
