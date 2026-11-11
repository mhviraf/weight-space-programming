"""Load the Wine dataset and z-score standardize it (deterministic, no randomness)."""
import csv
import hashlib
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wine.csv"
SHA256 = "2cce9d95d7d95165a591b84741d15d3300f6862a135a28e3601249d42f303056"
CLASS_NAMES = ["cultivar_0", "cultivar_1", "cultivar_2"]


# Fixed permutations used by the blind protocol, so that an agent who guesses the
# dataset cannot map remembered feature/class knowledge onto the input columns.
BLIND_SEED = 20261111


SPLIT_SEED = 7
TEST_FRACTION = 0.3


def _raw():
    raw = DATA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"data/wine.csv checksum mismatch: {digest}")
    rows = list(csv.reader(raw.decode().splitlines()))
    arr = np.array(rows[1:], dtype=float)
    return arr[:, :-1], arr[:, -1].astype(int), rows[0][:-1]


def load_holdout():
    """Blind + held-out protocol (experiment 3).

    Returns (X_feedback, y_feedback, X_test, y_test). Stratified 70/30 split with a
    fixed seed; standardization uses feedback-set statistics only; feature columns
    and class labels are permuted exactly as in the blind protocol."""
    X, y, _ = _raw()
    rng = np.random.default_rng(BLIND_SEED)
    fperm, cperm = rng.permutation(X.shape[1]), rng.permutation(3)
    X, y = X[:, fperm], cperm[y]
    srng = np.random.default_rng(SPLIT_SEED)
    test = np.zeros(len(y), dtype=bool)
    for c in range(3):
        idx = np.flatnonzero(y == c)
        test[srng.choice(idx, size=round(TEST_FRACTION * len(idx)), replace=False)] = True
    mu, sd = X[~test].mean(axis=0), X[~test].std(axis=0)
    X = (X - mu) / sd
    return X[~test], y[~test], X[test], y[test]


def for_state(st):
    """(X_feedback, y_feedback, X_test or None, y_test or None) for a run's state.json."""
    if st.get("holdout"):
        return load_holdout()
    X, y, _ = load(st.get("blind", False))
    return X, y, None, None


def load(blind=False):
    """Return (X_standardized [178x13], y [178], feature_names).

    blind=True shuffles the feature columns and relabels the classes with a
    fixed permutation (feature names are then hidden)."""
    raw = DATA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"data/wine.csv checksum mismatch: {digest}")
    rows = list(csv.reader(raw.decode().splitlines()))
    names = rows[0][:-1]
    arr = np.array(rows[1:], dtype=float)
    X, y = arr[:, :-1], arr[:, -1].astype(int)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    if blind:
        rng = np.random.default_rng(BLIND_SEED)
        fperm = rng.permutation(X.shape[1])
        cperm = rng.permutation(3)
        X, y = X[:, fperm], cperm[y]
        names = [f"x{i}" for i in range(X.shape[1])]
    return X, y, names
