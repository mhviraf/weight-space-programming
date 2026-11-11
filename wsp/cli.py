"""Command-line interface.

  python -m wsp init     --run DIR --seed S        random weights + pass 0 (baseline)
  python -m wsp forward  --run DIR                 one forward pass on DIR/weights.json
  python -m wsp describe                           data card (feature names, class means)
  python -m wsp random-search --run DIR --seed S   non-LLM baseline: 30 random-perturbation passes
  python -m wsp replay   --run DIR                 recompute every logged pass from snapshots
  python -m wsp report                             aggregate all runs -> results/
"""
import argparse
import csv
import json
import pathlib
import shutil
import sys
import time

import numpy as np

from . import data, model

QUOTA = 30
LOG_FIELDS = ["pass", "accuracy", "loss", "dead_h1", "dead_h2", "time"]


def _paths(run):
    run = pathlib.Path(run)
    return run, run / "weights.json", run / "log.csv", run / "history", run / "state.json"


def _record(run, w_text, metrics, step):
    run, _, log, hist, state = _paths(run)
    (hist / f"pass_{step:03d}.json").write_text(w_text)
    with log.open("a", newline="") as f:
        csv.writer(f).writerow([step, f"{metrics['accuracy']:.6f}", f"{metrics['loss']:.6f}",
                                metrics["dead_units"][0], metrics["dead_units"][1],
                                time.strftime("%Y-%m-%dT%H:%M:%S")])
    st = json.loads(state.read_text())
    st["passes_used"] = step
    state.write_text(json.dumps(st, indent=2) + "\n")


def _print(metrics, step):
    used = step
    print(f"pass {step}/{QUOTA}  (remaining: {QUOTA - used})")
    print(f"accuracy: {metrics['accuracy']:.4f}   cross-entropy loss: {metrics['loss']:.4f}")
    print("confusion matrix (rows = true class, cols = predicted):")
    for i, row in enumerate(metrics["confusion"]):
        print(f"  true {i}: " + "  ".join(f"{v:4d}" for v in row))
    print(f"dead ReLU units (never active on any sample): hidden1={metrics['dead_units'][0]}/10, "
          f"hidden2={metrics['dead_units'][1]}/10")


def cmd_init(a):
    run, wpath, log, hist, state = _paths(a.run)
    if run.exists() and any(run.iterdir()):
        sys.exit(f"{run} already exists; refusing to overwrite")
    hist.mkdir(parents=True)
    X, y, _ = data.load()
    w = model.init_weights(a.seed)
    text = model.dumps(w)
    wpath.write_text(text)
    state.write_text(json.dumps({"seed": a.seed, "quota": QUOTA, "passes_used": 0}, indent=2) + "\n")
    log.write_text(",".join(LOG_FIELDS) + "\n")
    m = model.evaluate(w, X, y)
    _record(run, text, m, 0)
    print(f"initialized {run} with seed {a.seed}; pass 0 is the random-weight baseline")
    _print(m, 0)


def cmd_forward(a):
    run, wpath, _, _, state = _paths(a.run)
    st = json.loads(state.read_text())
    if st["passes_used"] >= QUOTA:
        sys.exit(f"quota exhausted: {QUOTA} forward passes already used")
    text = wpath.read_text()
    try:
        w = json.loads(text)
        model.validate(w)
    except (ValueError, KeyError, TypeError) as e:
        sys.exit(f"invalid weights.json (no pass consumed): {e}")
    X, y, _ = data.load()
    m = model.evaluate(w, X, y)
    step = st["passes_used"] + 1
    _record(run, text, m, step)
    _print(m, step)


def cmd_describe(a):
    X, y, names = data.load()
    print(f"Wine dataset: {len(y)} samples, {X.shape[1]} features, 3 classes "
          f"(counts: {np.bincount(y).tolist()})")
    print("Features are z-score standardized (mean 0, std 1 over the whole dataset).")
    print("Per-class mean of each standardized feature (input index: name):")
    print(f"  {'idx':>3} {'feature':<30}" + "".join(f"{'class ' + str(c):>10}" for c in range(3)))
    for i, n in enumerate(names):
        print(f"  {i:>3} {n:<30}" + "".join(f"{X[y == c, i].mean():>10.2f}" for c in range(3)))


def cmd_random_search(a):
    """Non-LLM baseline under the same budget: perturb all weights with Gaussian
    noise, keep the change if loss improves (1+1 evolution strategy)."""
    cmd_init(argparse.Namespace(run=a.run, seed=a.seed))
    run, wpath, *_ = _paths(a.run)
    rng = np.random.default_rng(10_000 + a.seed)
    X, y, _ = data.load()
    best = model.load(wpath)
    best_loss = model.evaluate(best, X, y)["loss"]
    for step in range(1, QUOTA + 1):
        cand = {k: {"W": (np.asarray(v["W"]) + rng.normal(0, a.sigma, np.shape(v["W"]))).round(4).tolist(),
                    "b": (np.asarray(v["b"]) + rng.normal(0, a.sigma, np.shape(v["b"]))).round(4).tolist()}
                for k, v in best.items()}
        text = model.dumps(cand)
        m = model.evaluate(cand, X, y)
        _record(run, text, m, step)
        if m["loss"] < best_loss:
            best, best_loss = cand, m["loss"]
    wpath.write_text(model.dumps(best))
    print(f"random search done: final best loss {best_loss:.4f}")


def cmd_replay(a):
    run, _, log, hist, _ = _paths(a.run)
    X, y, _ = data.load()
    rows = list(csv.DictReader(log.open()))
    bad = 0
    for r in rows:
        w = model.load(hist / f"pass_{int(r['pass']):03d}.json")
        m = model.evaluate(w, X, y)
        ok = abs(m["accuracy"] - float(r["accuracy"])) < 1e-6 and abs(m["loss"] - float(r["loss"])) < 1e-5
        bad += not ok
        print(f"pass {r['pass']:>2}: acc {m['accuracy']:.4f} loss {m['loss']:.4f} {'OK' if ok else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


def cmd_report(a):
    from . import report
    report.main()


def main(argv=None):
    p = argparse.ArgumentParser(prog="wsp")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("--run", required=True); s.add_argument("--seed", type=int, required=True)
    s.set_defaults(f=cmd_init)
    s = sub.add_parser("forward"); s.add_argument("--run", required=True); s.set_defaults(f=cmd_forward)
    s = sub.add_parser("describe"); s.set_defaults(f=cmd_describe)
    s = sub.add_parser("random-search"); s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, required=True); s.add_argument("--sigma", type=float, default=0.1)
    s.set_defaults(f=cmd_random_search)
    s = sub.add_parser("replay"); s.add_argument("--run", required=True); s.set_defaults(f=cmd_replay)
    s = sub.add_parser("report"); s.set_defaults(f=cmd_report)
    a = p.parse_args(argv)
    a.f(a)
