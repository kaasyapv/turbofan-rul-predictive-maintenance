import numpy as np
from src import data, prep


def test_prep_round_trips_through_json(root, tmp_path):
    train, _, _ = data.load("FD001", root)
    p = data.fit(train, 2)
    prep.save(p, str(tmp_path / "prep.json"))
    back = prep.read(str(tmp_path / "prep.json"))
    cols = (train[data.COLS[2:5]], train[data.COLS[5:]])
    assert np.array_equal(prep.transform(p, *cols), prep.transform(back, *cols))


def test_constant_sensor_is_dropped(root):
    train, _, _ = data.load("FD001", root)
    assert 0 not in data.fit(train, 1)["keep"]


def test_no_engine_in_both_splits(root):
    train, _, _ = data.load("FD001", root)
    tr, va = data.split_units(train["unit"])
    assert not set(tr) & set(va)
    assert len(tr) + len(va) == train["unit"].nunique()


def test_preprocessing_is_fit_on_training_engines_only(root):
    train, _, _ = data.load("FD001", root)
    p, _, ids = data.build("FD001", root=root)
    only_tr = data.fit(train[train["unit"].isin(ids["train"])], 1)
    everyone = data.fit(train, 1)
    assert p["means"] == only_tr["means"]
    assert p["means"] != everyone["means"]


def test_test_set_is_one_window_per_engine(root):
    _, sets, _ = data.build("FD001", root=root)
    assert sets["te"]["x"].shape[:2] == (4, 30)
    assert len(sets["te"]["y"]) == 4
