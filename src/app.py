"""Fleet view. Run with: streamlit run src/app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import data  # noqa: E402
from src.metrics import rmse  # noqa: E402
from src.predict import load, run  # noqa: E402
from src.prep import transform, windows  # noqa: E402


@st.cache_data
def fleet(name: str) -> pd.DataFrame:
    _, test, true = data.load(name)
    prep, sessions = load(name)
    x = transform(prep, test[data.COLS[2:5]], test[data.COLS[5:]])
    p = run(sessions, windows(x, test["unit"], last_only=True))
    return pd.DataFrame({"engine": test["unit"].unique(), "predicted": p.mean(0), "spread": p.std(0),
                         "true": true, "cycles_seen": test.groupby("unit", sort=False)["cycle"].max().to_numpy()})


name = st.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"])
limit = st.slider("Flag engines with predicted RUL at or under", 5, 100, 30)
df = fleet(name).sort_values("predicted").reset_index(drop=True)
df["flagged"] = df["predicted"] <= limit

a, b, c = st.columns(3)
a.metric("Engines", len(df))
b.metric("Flagged", int(df["flagged"].sum()))
c.metric("Test RMSE", f"{rmse(df['predicted'], df['true']):.2f} cycles")
st.dataframe(df, hide_index=True)
st.scatter_chart(df, x="true", y="predicted")
