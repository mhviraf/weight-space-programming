"""Load the Wine dataset and z-score standardize it (deterministic, no randomness)."""
import csv
import hashlib
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wine.csv"
SHA256 = "2cce9d95d7d95165a591b84741d15d3300f6862a135a28e3601249d42f303056"
CLASS_NAMES = ["cultivar_0", "cultivar_1", "cultivar_2"]


def load():
    """Return (X_standardized [178x13], y [178], feature_names)."""
    raw = DATA.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"data/wine.csv checksum mismatch: {digest}")
    rows = list(csv.reader(raw.decode().splitlines()))
    names = rows[0][:-1]
    arr = np.array(rows[1:], dtype=float)
    X, y = arr[:, :-1], arr[:, -1].astype(int)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y, names
