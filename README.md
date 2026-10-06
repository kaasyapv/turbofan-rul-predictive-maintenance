# turbofan-rul-predictive-maintenance

Predicts how many flight cycles a jet engine has left before failure from its last 30 cycles of sensor readings, serves the model as an API, and ranks a fleet of engines so the ones closest to failure get pulled for maintenance first. It uses the NASA C-MAPSS simulated turbofan data, all four subsets.

## Headline

Every model beats the constant baseline by a wide margin on all four subsets. XGBoost on window features is as good as the LSTM or better on three of them, and the LSTM ensemble only wins on FD003. On FD004 the two are 0.14 cycles apart, which is inside the noise of one split. The seed spread correlates only weakly with absolute error, from -0.07 to 0.22, so it is a rough hint at best. Capping the target helped XGBoost on FD001 and FD003 and made no real difference on FD004, but on FD002 the raw target did better.

<!-- results -->
Test RMSE in cycles, last window of each test engine.

| Subset | Constant | XGBoost | LSTM mean +- std over 3 seeds | LSTM ensemble |
|---|---|---|---|---|
| FD001 | 42.94 | 14.11 | 15.51 +- 0.93 | 14.89 |
| FD002 | 54.06 | 25.10 | 27.39 +- 0.42 | 27.07 |
| FD003 | 44.95 | 14.99 | 14.90 +- 0.65 | 14.31 |
| FD004 | 54.89 | 26.18 | 26.59 +- 0.19 | 26.32 |

Flag means predicted RUL of 30 or less, scored against engines whose true RUL is 30 or less.

| Subset | Model | NASA score | Flag precision | Flag recall |
|---|---|---|---|---|
| FD001 | XGBoost | 318 | 0.95 | 0.72 |
| FD001 | LSTM ensemble | 417 | 0.91 | 0.84 |
| FD002 | XGBoost | 6045 | 0.97 | 0.97 |
| FD002 | LSTM ensemble | 10383 | 0.97 | 0.97 |
| FD003 | XGBoost | 425 | 0.90 | 0.90 |
| FD003 | LSTM ensemble | 353 | 0.90 | 0.90 |
| FD004 | XGBoost | 4527 | 0.96 | 0.87 |
| FD004 | LSTM ensemble | 4307 | 0.94 | 0.83 |

Ablation and seed spread.

| Subset | XGBoost capped target | XGBoost raw target | Seed spread vs abs error (corr) |
|---|---|---|---|
| FD001 | 14.11 | 15.70 | 0.22 |
| FD002 | 25.10 | 24.09 | 0.18 |
| FD003 | 14.99 | 17.29 | 0.18 |
| FD004 | 26.18 | 26.20 | -0.07 |
<!-- end results -->

The tables are generated from the `results_FD00x.json` files by `python -m src.tables`.

## Run it

```
pip install -r requirements.txt
./download_data.sh && ./check_data.sh
python -m src.train && python -m src.export
uvicorn src.serve:app
streamlit run src/app.py
```

Training all four subsets takes a few minutes per subset on a laptop CPU. `python -m pytest` runs the tests.

On macOS, XGBoost needs the OpenMP runtime. If `brew install libomp` is not an option, point `DYLD_FALLBACK_LIBRARY_PATH` at the `torch/lib` folder inside your virtualenv.

Example request:

```
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"subset": "FD001",
  "readings": [{"cycle": 1, "settings": [0.0, 0.0, 100.0], "sensors": [<21 numbers>]}]}'
```

The response holds `rul_cycles` (mean of the three LSTM seeds, clipped to 0 and 125), `model_spread`, `maintenance_flag` (true at 30 or less) and `cycles_seen`. An empty history, a wrong number of settings or sensors, or an unknown subset returns 422.

## How it works

* Target is the last cycle of an engine minus the current cycle, capped at 125 for training. The test score uses the true values from the RUL files.
* FD002 and FD004 have six operating conditions. KMeans on the three settings assigns each row to one, and every sensor is standardised inside its condition. Sensors that never move are dropped.
* 20 percent of the training engines are held out for early stopping. Preprocessing is fit on the other 80 percent only.
* Each sample is a window of 30 cycles. Engines with fewer cycles are front padded with their first row.
* XGBoost runs on last, mean, std, min, max and slope of each sensor plus the cycle count. The LSTM has 2 layers of 64 units. Three seeds are trained per subset.
* Predictions are clipped to 0 and 125 before scoring. The NASA score penalises late predictions more than early ones.
* `src/export.py` writes ONNX models (opset 17) and `prep.json`, so the API needs only numpy and ONNX Runtime.

## Data

Data is the Turbofan Engine Degradation Simulation Data Set (C-MAPSS) from the NASA Prognostics Center of Excellence. `download_data.sh` tries the official NASA zip first and falls back to a GitHub mirror. The zip worked when the results above were made. `check_data.sh` compares the line counts of all 12 files with the official release. The `data/` folder is not committed.

Saxena, Goebel, Simon and Eklund, "Damage Propagation Modeling for Aircraft Engine Prognostics", PHM 2008.

## Limitations

* The data is simulated, not real fleet data. Real engines have noisier sensors, missing data and maintenance events that this set does not model.
* The spread across the three LSTM seeds is not a calibrated uncertainty. See the correlation column above for how loosely it tracks the error.
* Test scores come from one split of training engines and one run per model, so differences of a cycle or so between models are within what a different split would change.
* The Dockerfile for the API has not been built, because no Docker daemon was running where this was developed.
* The Streamlit app was run and driven by script, but it has not been checked by eye in a browser.

## License

MIT
