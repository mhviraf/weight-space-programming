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
