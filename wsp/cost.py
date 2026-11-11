"""Cost vs. performance: price each run from results-*/usage.csv and plot it.

Prices: USD per million tokens, Anthropic first-party API list prices as of 2026-11.
Cache writes were all 5-minute TTL, billed at 1.25x the input price. The Haiku 5.5
cache-read price is assumed to be 0.1x input, the same ratio as the other two models.
All Haiku prompts were under 100K tokens, so the standard Haiku tier applies.
"""
import csv
import pathlib

import numpy as np

PRICES = {  # input, output, cache read (per 1M tokens)
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
CACHE_WRITE_MULT = 1.25
MODELS = ["haiku", "sonnet", "opus"]
EFFORTS = ["low", "medium", "high"]
COLORS = {"haiku": "#2a78d6", "sonnet": "#eb6834", "opus": "#1baf7a"}


def run_cost(r):
    pin, pout, pread = PRICES[r["model_id"]]
    return (int(r["input_tokens"]) * pin + int(r["cache_write_tokens"]) * pin * CACHE_WRITE_MULT
            + int(r["cache_read_tokens"]) * pread + int(r["output_tokens"]) * pout) / 1e6


def main(results, title):
    out = pathlib.Path(results)
    usage = list(csv.DictReader((out / "usage.csv").open()))
    cost = {(r["config"], int(r["trial"])): run_cost(r) for r in usage}
    final = {}
    for r in csv.DictReader((out / "all_passes.csv").open()):
        final[(r["config"], int(r["trial"]))] = float(r["accuracy"])  # rows are in pass order
    rows = []
    for (cfg, t), c in sorted(cost.items()):
        rows.append({"config": cfg, "trial": t, "cost_usd": round(c, 4), "accuracy": final[(cfg, t)]})
    with (out / "cost.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    held = "held-out " if "holdout" in results else ""
    rs = [v for (c, _), v in final.items() if c == "random-search"]
    if rs:
        ax.axhline(np.mean(rs), color="#8a8985", lw=1.2, ls="--")
        ax.text(0.99, np.mean(rs) + 0.008, f"random search, no LLM, $0 ({np.mean(rs):.1%})",
                fontsize=8, color="#52514e", ha="right", transform=ax.get_yaxis_transform())
    summary = ["| model | effort | mean cost per run (USD) | mean accuracy |", "|---|---|---|---|"]
    for m in MODELS:
        xs, ys = [], []
        for e in EFFORTS:
            pts = [(r["cost_usd"], r["accuracy"]) for r in rows if r["config"] == f"{m}-{e}"]
            if not pts:
                continue
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=14, color=COLORS[m], alpha=0.3,
                       edgecolors="none")
            xs.append(cx); ys.append(cy)
            ax.annotate(e, (cx, cy), textcoords="offset points", xytext=(7, -3), fontsize=8,
                        color="#52514e")
            summary.append(f"| {m} | {e} | {cx:.3f} | {cy:.3f} |")
        ax.plot(xs, ys, "-o", color=COLORS[m], lw=2, ms=8, mec="white", mew=1.5,
                label=f"{m.capitalize()} (mean of 3 seeds)")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 3)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_xlabel("API cost per run, USD (log scale; 30 forward passes)")
    ax.set_ylabel(f"{held}accuracy after pass 30")
    ax.set_ylim(0.25, 1.0)
    ax.grid(color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"{title}\ncost vs. {held}accuracy (faint dots = individual seeds)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "cost_vs_accuracy.png")
    plt.close(fig)
    (out / "cost_summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
