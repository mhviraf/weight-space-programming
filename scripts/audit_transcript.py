"""Audit an agent transcript (Claude Code JSONL) for protocol violations.

Allowed tool calls:
  Read  RUN/prompt.md | RUN/weights.json
  Edit/Write  RUN/weights.json
  Bash  exactly: [cd REPO && ]python -m wsp forward --run RUN
  SubagentHandback (final report)
Everything else is flagged. Writes a compact tool-call log to RUN/audit.txt.

Usage: python scripts/audit_transcript.py TRANSCRIPT.jsonl runs-blind/haiku-medium/trial0
"""
import json
import pathlib
import sys

repo = pathlib.Path(__file__).resolve().parents[1]
transcript, run = pathlib.Path(sys.argv[1]), (repo / sys.argv[2]).resolve()
fwd = f"python -m wsp forward --run {run}"
forward_ok = {fwd, f"cd {repo} && {fwd}"}
ok_read = {str(run / "prompt.md"), str(run / "weights.json")}
ok_write = {str(run / "weights.json")}

calls, bad = [], []
for line in transcript.read_text().splitlines():
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        continue
    content = (ev.get("message") or {}).get("content")
    if not isinstance(content, list):
        continue
    for c in content:
        if not isinstance(c, dict) or c.get("type") != "tool_use":
            continue
        name, inp = c.get("name"), c.get("input", {})
        if name == "Read":
            arg, good = inp.get("file_path"), inp.get("file_path") in ok_read
        elif name in ("Edit", "Write", "MultiEdit"):
            arg, good = inp.get("file_path"), inp.get("file_path") in ok_write
        elif name == "Bash":
            arg, good = inp.get("command"), inp.get("command", "").strip() in forward_ok
        elif name.endswith("SubagentHandback"):  # the agent's final report
            arg, good = "(final report)", True
        else:
            arg, good = json.dumps(inp)[:200], False
        calls.append(f"{'OK ' if good else 'BAD'} {name:<6} {arg}")
        if not good:
            bad.append(calls[-1])

counts = {}
for c in calls:
    k = c.split()[1]
    counts[k] = counts.get(k, 0) + 1
summary = [f"transcript: {transcript.name}", f"tool calls: {len(calls)} {counts}",
           f"violations: {len(bad)}"] + [f"  {b}" for b in bad]
(run / "audit.txt").write_text("\n".join(summary + ["", "full call log:"] + calls) + "\n")
print("\n".join(summary))
sys.exit(1 if bad else 0)
