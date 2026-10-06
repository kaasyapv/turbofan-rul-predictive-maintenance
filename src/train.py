"""Train and score every model on one or more C-MAPSS subsets."""
import argparse
import json
import os
import numpy as np
import torch
import xgboost as xgb
from . import data, model
from .features import features
from .metrics import clip, report
from .prep import save


def boost(f, y, fv, yv):
    m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=50, random_state=0, n_jobs=-1)
    return m.fit(f, y, eval_set=[(fv, yv)], verbose=False)


def run(name: str):
    print(f"== {name}")
    prep, s, ids = data.build(name)
    tr, va, te = s["tr"], s["va"], s["te"]
    true = te["y"]
    f, fv, ft = (features(d["x"], d["cycle"]) for d in (tr, va, te))
    res = {"subset": name, "train_engines": len(ids["train"]), "val_engines": len(ids["val"]),
           "test_engines": len(true), "kept_sensors": len(prep["keep"])}

    res["baseline"] = report(np.full(len(true), tr["y"].mean()), true)
    capped = boost(f, tr["y"], fv, va["y"])
    res["xgb"] = report(capped.predict(ft), true) | {"trees": capped.best_iteration + 1}
    raw = boost(f, tr["y_raw"], fv, va["y_raw"])
    res["xgb_raw"] = report(raw.predict(ft), true) | {"trees": raw.best_iteration + 1}
    print("  xgb", res["xgb"]["rmse"], "raw", res["xgb_raw"]["rmse"])

    os.makedirs(f"models/{name}", exist_ok=True)
    preds = []
    for seed in range(3):
        net = model.fit(tr["x"], tr["y"], va["x"], va["y"], seed)
        torch.save(net.state_dict(), f"models/{name}/lstm_{seed}.pt")
        preds.append(clip(model.predict(net, te["x"])))
    preds = np.stack(preds)
    runs = [report(p, true) for p in preds]
    res["lstm_seeds"] = runs
    res["lstm_mean"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    res["lstm_std"] = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    res["lstm_ensemble"] = report(preds.mean(0), true)
    # does disagreement between seeds point at the engines we get wrong
    spread, err = preds.std(0), np.abs(preds.mean(0) - true)
    res["spread_error_corr"] = float(np.corrcoef(spread, err)[0, 1])
    print("  lstm ensemble", res["lstm_ensemble"]["rmse"])

    save(prep, f"models/{name}/prep.json")
    with open(f"results_{name}.json", "w") as out:
        json.dump(res, out, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="*", default=["FD001", "FD002", "FD003", "FD004"])
    for name in ap.parse_args().subsets:
        run(name)
