import numpy as np
import pytest


@pytest.fixture
def root(tmp_path):
    """Ten tiny engines per file in the NASA text layout, sensor 1 constant."""
    rng = np.random.default_rng(0)

    def rows(n_units, lo, hi):
        out = []
        for u in range(1, n_units + 1):
            for c in range(1, rng.integers(lo, hi) + 1):
                out.append([u, c, *rng.normal(size=3), 5.0, *rng.normal(u, 1, size=20)])
        return np.array(out)

    np.savetxt(tmp_path / "train_FD001.txt", rows(10, 15, 60))
    np.savetxt(tmp_path / "test_FD001.txt", rows(4, 15, 60))
    np.savetxt(tmp_path / "RUL_FD001.txt", [10, 20, 30, 40])
    return str(tmp_path)
