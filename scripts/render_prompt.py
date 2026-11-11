"""Render prompts/agent_prompt.md for one run and save it as RUN/prompt.md.

Usage: python scripts/render_prompt.py runs/opus-high/trial0
"""
import pathlib
import subprocess
import sys

repo = pathlib.Path(__file__).resolve().parents[1]
run = (repo / sys.argv[1]).resolve()
card = subprocess.run([sys.executable, "-m", "wsp", "describe"], cwd=repo,
                      capture_output=True, text=True, check=True).stdout
text = ((repo / "prompts" / "agent_prompt.md").read_text()
        .replace("{RUN}", str(run)).replace("{REPO}", str(repo)).replace("{DATA_CARD}", card))
(run / "prompt.md").write_text(text)
print(text)
