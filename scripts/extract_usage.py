"""Extract per-run token usage from agent transcripts into a CSV (committed).

Transcripts are Claude Code subagent JSONL files; they are not in the repo, so this
step is not reproducible from the repo alone - its output (results-*/usage.csv) is.
Streamed responses repeat the same message id; each id is counted once, using the
largest output_tokens seen for it.

Usage: python scripts/extract_usage.py MAP.txt TRANSCRIPT_DIR OUT.csv
  MAP.txt lines: "<transcript stem> <run dir>"
"""
import csv
import json
import pathlib
import sys

mp, tdir, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
rows = []
for line in open(mp):
    stem, run = line.split()
    msgs = {}
    for l in open(tdir / f"{stem}.output"):
        m = json.loads(l).get("message") or {}
        if m.get("role") != "assistant" or not m.get("usage"):
            continue
        u, prev = m["usage"], msgs.get(m["id"])
        if prev is None or u.get("output_tokens", 0) > prev["output_tokens"]:
            msgs[m["id"]] = {"model": m.get("model"),
                             "input_tokens": u.get("input_tokens", 0),
                             "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
                             "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                             "output_tokens": u.get("output_tokens", 0)}
    models = {v["model"] for v in msgs.values()}
    assert len(models) == 1, (run, models)
    cfg, trial = run.split("/")[1], run.split("/")[2]
    tot = {k: sum(v[k] for v in msgs.values())
           for k in ("input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens")}
    rows.append({"config": cfg, "trial": int(trial[5:]), "model_id": models.pop(),
                 "api_calls": len(msgs), **tot})
rows.sort(key=lambda r: (r["config"], r["trial"]))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}: {len(rows)} runs")
