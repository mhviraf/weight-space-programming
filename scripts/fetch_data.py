"""Export the UCI Wine dataset (bundled with scikit-learn) to data/wine.csv.

This is only needed once; the CSV is committed so the rest of the repo depends on
numpy only. The SHA-256 of the CSV is checked by wsp/data.py.

Source: Forina et al., UCI Machine Learning Repository, "Wine" (1991).
178 wines from three Italian cultivars, 13 chemical-analysis features.
"""
import csv
import pathlib

from sklearn.datasets import load_wine

out = pathlib.Path(__file__).resolve().parents[1] / "data" / "wine.csv"
ds = load_wine()
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(list(ds.feature_names) + ["target"])
    for x, y in zip(ds.data, ds.target):
        w.writerow([repr(float(v)) for v in x] + [int(y)])
print(f"wrote {out}")
