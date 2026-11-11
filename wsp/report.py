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
import json
import pathlib

import numpy as np

from . import data, model

ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFORTS = ["low", "medium", "high"]
# validated categorical slots (light mode) + neutral gray for the baseline
COLORS = {"low": "#2a78d6", "medium": "#1baf7a", "high": "#eb6834", "random-search": "#8a8985"}


def _flat(w):
    return np.concatenate([np.ravel(w[l][k]) for l in model.LAYERS for k in ("W", "b")])


def collect(RUNS):
    import json
    rows = []
    for cfg_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        for trial_dir in sorted(cfg_dir.glob("trial*")):
            prev = None
            st = json.loads((trial_dir / "state.json").read_text())
            Xf, yf, Xt, yt = data.for_state(st)
            # holdout protocol: the scored metric is held-out accuracy; feedback accuracy kept separately
            X, y = (Xt, yt) if Xt is not None else (Xf, yf)
            for snap in sorted((trial_dir / "history").glob("pass_*.json")):
                w = model.load(snap)
                m = model.evaluate(w, X, y)
                flat = _flat(w)
                changed = 0 if prev is None else int((np.abs(flat - prev) > 1e-12).sum())
                prev = flat
                rows.append({"config": cfg_dir.name, "trial": int(trial_dir.name[5:]),
                             "pass": int(snap.stem[5:]), "accuracy": m["accuracy"],
                             "loss": m["loss"], "weights_changed": changed,
                             "feedback_accuracy": model.evaluate(w, Xf, yf)["accuracy"]})
    return rows


def curves(rows, cfg, key):
    trials = sorted({r["trial"] for r in rows if r["config"] == cfg})
    out = []
    for t in trials:
        rs = sorted((r for r in rows if r["config"] == cfg and r["trial"] == t), key=lambda r: r["pass"])
        out.append([r[key] for r in rs])
    return out


def centroid_accuracy():
    """Accuracy of 'predict the nearest class mean' using only the data-card means."""
    X, y, _ = data.load()
    M = np.stack([X[y == c].mean(0) for c in range(3)])
    return float((((X[:, None, :] - M) ** 2).sum(-1).argmin(1) == y).mean())


def plot(rows, key, ylabel, path, MODELS, title, show_ref=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.4 * len(MODELS) + 0.4, 4.2), sharey=True, dpi=150, squeeze=False)
    axes = axes[0]
    for ax, mname in zip(axes, MODELS):
        series = [("random-search", "random search (1+1 ES)")] + \
                 [(f"{mname}-{e}", f"{mname} · {e} effort") for e in EFFORTS]
        for cfg, label in series:
            cs = curves(rows, cfg, key)
            if not cs:
                continue
            color = COLORS["random-search" if cfg == "random-search" else cfg.split("-")[1]]
            n = max(len(c) for c in cs)
            for c in cs:
                ax.plot(range(len(c)), c, color=color, lw=0.8, alpha=0.3)
                if len(c) < n:  # run stopped before using its full quota
                    ax.plot(len(c) - 1, c[-1], "x", color=color, ms=6)
            padded = np.array([c + [np.nan] * (n - len(c)) for c in cs])
            mean = np.nanmean(padded, axis=0)  # mean over the trials that reached that pass
            ax.plot(range(n), mean, color=color, lw=2, label=f"{label} (mean of {len(cs)})")
        if key == "accuracy" and show_ref:
            ref = centroid_accuracy()
            ax.axhline(ref, color="#52514e", lw=0.8, ls="--")
            ax.text(30, ref - 0.035, f"nearest class mean from data card: {ref:.1%}",
                    ha="right", fontsize=7, color="#52514e")
        ax.set_title(mname.capitalize(), fontsize=11, loc="left")
        ax.set_xlabel("forward pass (0 = random weights)")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    axes[0].set_ylabel(ylabel)
    if key.endswith("accuracy"):
        axes[0].set_ylim(0, 1.02)
    import textwrap
    full = f"{title}: {ylabel} after every forward pass (thin = individual trials, x = run stopped early)"
    fig.suptitle(textwrap.fill(full, width=int(40 * len(MODELS) + 10)), fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(runs="runs", out="results", MODELS=("haiku", "sonnet", "opus"), title="Hand-edited MLP on UCI Wine"):
    RUNS, OUT = ROOT / runs, ROOT / out
    OUT.mkdir(exist_ok=True)
    rows = collect(RUNS)
    with (OUT / "all_passes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "accuracy": f"{r['accuracy']:.6f}", "loss": f"{r['loss']:.6f}",
                        "feedback_accuracy": f"{r['feedback_accuracy']:.6f}"})
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
    # Strategy proxy: how much of the hidden layers did each run rewrite (pass 0 -> last pass)?
    lines += ["", "| run | final acc | hidden1 weights changed | hidden2 weights changed | output weights changed |",
              "|---|---|---|---|---|"]
    for cfg_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name != "random-search"):
        for trial_dir in sorted(cfg_dir.glob("trial*")):
            snaps = sorted((trial_dir / "history").glob("pass_*.json"))
            a, b = model.load(snaps[0]), model.load(snaps[-1])
            ch = {l: int((np.asarray(a[l]["W"]) != np.asarray(b[l]["W"])).sum()) for l in model.LAYERS}
            acc = curves(rows, cfg_dir.name, "accuracy")[int(trial_dir.name[5:])][-1]
            lines.append(f"| {cfg_dir.name}/{trial_dir.name} | {acc:.3f} | {ch['hidden1']}/130 | "
                         f"{ch['hidden2']}/100 | {ch['output']}/30 |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    show_ref = runs == "runs"
    holdout = any(json.loads(p.read_text()).get("holdout") for p in RUNS.glob("*/trial*/state.json"))
    label = "held-out accuracy" if holdout else "accuracy"
    plot(rows, "accuracy", label, OUT / "accuracy.png", MODELS, title, show_ref)
    plot(rows, "loss", "held-out cross-entropy loss" if holdout else "cross-entropy loss",
         OUT / "loss.png", MODELS, title, show_ref)
    if holdout:  # what the agents actually saw
        plot(rows, "feedback_accuracy", "feedback-set accuracy (what agents saw)",
             OUT / "feedback_accuracy.png", MODELS, title, False)
    print(f"wrote {OUT}/all_passes.csv, summary.md and charts")
