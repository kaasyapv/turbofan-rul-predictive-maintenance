"""Load C-MAPSS, split by engine, standardise per operating condition, cut windows."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .prep import nearest, transform, windows

CAP = 125
COLS = ["unit", "cycle"] + [f"set{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]


def load(name: str, root: str = "data"):
    """Return train df, test df and the true RUL of each test engine."""
    train = pd.read_csv(f"{root}/train_{name}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(f"{root}/test_{name}.txt", sep=r"\s+", header=None, names=COLS)
    rul = np.loadtxt(f"{root}/RUL_{name}.txt")
    return train, test, rul


def split_units(units, frac=0.2, seed=0):
    """Hold out whole engines, never individual cycles."""
    ids = np.unique(units)
    rng = np.random.default_rng(seed)
    val = rng.choice(ids, size=round(len(ids) * frac), replace=False)
    return np.setdiff1d(ids, val), np.sort(val)


def rul_of(df, cap=None):
    """Last cycle of the engine minus this cycle."""
    rul = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return rul.clip(upper=cap).to_numpy(float) if cap else rul.to_numpy(float)


def fit(df, k: int, seed=0) -> dict:
    """Cluster the settings, then take sensor mean and std inside each cluster."""
    settings, sensors = df[COLS[2:5]].to_numpy(), df[COLS[5:]].to_numpy()
    centers = KMeans(k, n_init=10, random_state=seed).fit(settings).cluster_centers_
    label = nearest(centers, settings)
    means = np.stack([sensors[label == c].mean(0) for c in range(k)])
    stds = np.stack([sensors[label == c].std(0) for c in range(k)])
    # a sensor that never moves inside any cluster carries no signal
    keep = np.where((stds > 1e-6).all(0))[0]
    return {"centers": centers.tolist(), "means": means[:, keep].tolist(),
            "stds": stds[:, keep].tolist(), "keep": keep.tolist()}












def build(name: str, seed=0, root="data"):
    """Everything train.py needs for one subset."""
    train, test, rul = load(name, root)
    tr_ids, va_ids = split_units(train["unit"], seed=seed)
    tr, va = train[train["unit"].isin(tr_ids)], train[train["unit"].isin(va_ids)]
    prep = fit(tr, 6 if name in ("FD002", "FD004") else 1, seed)
    sets = {}
    for tag, df in (("tr", tr), ("va", va)):
        x = transform(prep, df[COLS[2:5]], df[COLS[5:]])
        u = df["unit"].to_numpy()
        sets[tag] = dict(x=windows(x, u), cycle=df["cycle"].to_numpy(float),
                         y=rul_of(df, CAP), y_raw=rul_of(df))
    # test engines stop mid life, so only the final window is scored
    x = transform(prep, test[COLS[2:5]], test[COLS[5:]])
    u = test["unit"].to_numpy()
    sets["te"] = dict(x=windows(x, u, last_only=True), cycle=test.groupby("unit")["cycle"].max().to_numpy(float), y=rul)
    return prep, sets, dict(train=tr_ids, val=va_ids)
