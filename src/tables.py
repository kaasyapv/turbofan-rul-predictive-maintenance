"""Rewrite the results tables in README.md from the results_*.json files."""
import json

NAMES = ["FD001", "FD002", "FD003", "FD004"]
START, END = "<!-- results -->", "<!-- end results -->"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def build() -> str:
    r = {n: json.load(open(f"results_{n}.json")) for n in NAMES}
    f = lambda v, d=2: f"{v:.{d}f}"
    main = table(["Subset", "Constant", "XGBoost", "LSTM mean +- std over 3 seeds", "LSTM ensemble"], [
        [n, f(x["baseline"]["rmse"]), f(x["xgb"]["rmse"]),
         f"{f(x['lstm_mean']['rmse'])} +- {f(x['lstm_std']['rmse'])}", f(x["lstm_ensemble"]["rmse"])]
        for n, x in r.items()])
    flag = table(["Subset", "Model", "NASA score", "Flag precision", "Flag recall"], [
        [n, m, f(x[k]["score"], 0), f(x[k]["precision"]), f(x[k]["recall"])]
        for n, x in r.items() for m, k in (("XGBoost", "xgb"), ("LSTM ensemble", "lstm_ensemble"))])
    extra = table(["Subset", "XGBoost capped target", "XGBoost raw target", "Seed spread vs abs error (corr)"], [
        [n, f(x["xgb"]["rmse"]), f(x["xgb_raw"]["rmse"]), f(x["spread_error_corr"])] for n, x in r.items()])
    return (f"Test RMSE in cycles, last window of each test engine.\n\n{main}\n\n"
            f"Flag means predicted RUL of 30 or less, scored against engines whose true RUL is 30 or less.\n\n{flag}\n\n"
            f"Ablation and seed spread.\n\n{extra}")


if __name__ == "__main__":
    text = open("README.md").read()
    head, rest = text.split(START)
    tail = rest.split(END)[1]
    open("README.md", "w").write(f"{head}{START}\n{build()}\n{END}{tail}")
    print("README tables updated")
