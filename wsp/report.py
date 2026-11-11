"""Aggregate every run under runs/ into results/.

Outputs
  results/all_passes.csv  one row per (config, trial, pass): accuracy, loss, #weights changed
  results/summary.md      final / best accuracy per config
  results/accuracy.png    accuracy vs. forward pass, one panel per model
  results/loss.png        same for cross-entropy loss
Metrics are re-computed from the saved weight snapshots (not read from the logs),
so this doubles as a reproducibility check.
"""
import csv
import pathlib

import numpy as np

from . import data, model

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "results"
MODELS = ["haiku", "sonnet", "opus"]
EFFORTS = ["low", "high"]
# validated categorical slots (light mode) + neutral gray for the baseline
COLORS = {"low": "#2a78d6", "high": "#eb6834", "random-search": "#8a8985"}


def _flat(w):
    return np.concatenate([np.ravel(w[l][k]) for l in model.LAYERS for k in ("W", "b")])


def collect():
    X, y, _ = data.load()
    rows = []
    for cfg_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        for trial_dir in sorted(cfg_dir.glob("trial*")):
            prev = None
            for snap in sorted((trial_dir / "history").glob("pass_*.json")):
                w = model.load(snap)
                m = model.evaluate(w, X, y)
                flat = _flat(w)
                changed = 0 if prev is None else int((np.abs(flat - prev) > 1e-12).sum())
                prev = flat
                rows.append({"config": cfg_dir.name, "trial": int(trial_dir.name[5:]),
                             "pass": int(snap.stem[5:]), "accuracy": m["accuracy"],
                             "loss": m["loss"], "weights_changed": changed})
    return rows


def curves(rows, cfg, key):
    trials = sorted({r["trial"] for r in rows if r["config"] == cfg})
    out = []
    for t in trials:
        rs = sorted((r for r in rows if r["config"] == cfg and r["trial"] == t), key=lambda r: r["pass"])
        out.append([r[key] for r in rs])
    return out


def plot(rows, key, ylabel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, dpi=150)
    for ax, mname in zip(axes, MODELS):
        series = [("random-search", "random search (1+1 ES)")] + \
                 [(f"{mname}-{e}", f"{mname} · {e} effort") for e in EFFORTS]
        for cfg, label in series:
            cs = curves(rows, cfg, key)
            if not cs:
                continue
            color = COLORS["random-search" if cfg == "random-search" else cfg.split("-")[1]]
            n = min(len(c) for c in cs)
            for c in cs:
                ax.plot(range(len(c)), c, color=color, lw=0.8, alpha=0.3)
            mean = np.mean([c[:n] for c in cs], axis=0)
            ax.plot(range(n), mean, color=color, lw=2, label=f"{label} (mean of {len(cs)})")
        ax.set_title(mname.capitalize(), fontsize=11, loc="left")
        ax.set_xlabel("forward pass (0 = random weights)")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    axes[0].set_ylabel(ylabel)
    if key == "accuracy":
        axes[0].set_ylim(0, 1.02)
    fig.suptitle(f"Hand-edited MLP on UCI Wine: {ylabel} after every forward pass "
                 "(thin = individual trials)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows = collect()
    with (OUT / "all_passes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "accuracy": f"{r['accuracy']:.6f}", "loss": f"{r['loss']:.6f}"})
    lines = ["| config | trials | pass 0 acc | final acc (mean) | best acc (mean) | final accs | passes used |",
             "|---|---|---|---|---|---|---|"]
    order = ["random-search"] + [f"{m}-{e}" for m in MODELS for e in EFFORTS]
    for cfg in order:
        acc = curves(rows, cfg, "accuracy")
        if not acc:
            continue
        lines.append(f"| {cfg} | {len(acc)} | {np.mean([c[0] for c in acc]):.3f} | "
                     f"{np.mean([c[-1] for c in acc]):.3f} | {np.mean([max(c) for c in acc]):.3f} | "
                     f"{', '.join(f'{c[-1]:.3f}' for c in acc)} | "
                     f"{', '.join(str(len(c) - 1) for c in acc)} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(rows, "accuracy", "accuracy", OUT / "accuracy.png")
    plot(rows, "loss", "cross-entropy loss", OUT / "loss.png")
    print(f"wrote {OUT}/all_passes.csv, summary.md, accuracy.png, loss.png")
