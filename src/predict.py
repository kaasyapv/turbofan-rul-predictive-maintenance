"""ONNX inference for the three seed models of a subset."""
import numpy as np
import onnxruntime as ort
from .prep import read, transform, windows


def load(name: str, root: str = "models"):
    """Preprocessing stats and one ONNX session per seed."""
    prep = read(f"{root}/{name}/prep.json")
    return prep, [ort.InferenceSession(f"{root}/{name}/lstm_{s}.onnx") for s in range(3)]


def run(sessions, w):
    """Clipped RUL from each seed, shape (3, n)."""
    w = np.asarray(w, np.float32)
    return np.clip([s.run(None, {"x": w})[0] * 125 for s in sessions], 0, 125)


def predict(prep: dict, sessions, readings):
    """Ensemble mean and seed spread for one engine. Rows are cycle, 3 settings, 21 sensors."""
    a = np.asarray(readings, float)
    x = transform(prep, a[:, 1:4], a[:, 4:25])
    p = run(sessions, windows(x, np.zeros(len(x)), last_only=True))[:, 0]
    return float(p.mean()), float(p.std())
