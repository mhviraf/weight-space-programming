# weight-space-programming

Can an LLM improve a small neural network **only by editing its weights by hand
and running forward passes**, with no training, no gradients and no optimizer code?

This repo runs that test on a tiny MLP and real data. It compares Claude Haiku 5.5,
Sonnet 5.5 and Opus 5.5 at several effort levels against a no-LLM random-search
baseline with the same budget. Everything except the LLM's own choices is
deterministic and can be replayed.

![cost vs held-out accuracy](results-holdout/cost_vs_accuracy.png)

## TL;DR

- **Setup.** An MLP `13 → 10 → 10 → 3` on the UCI Wine dataset starts from random weights. Each agent gets 30 forward passes. It may only read `weights.json`, edit it by hand, and run one forward-pass command.
- **With no information about the data** (experiment 3: shuffled inputs and labels, accuracy-only feedback), the score is held-out accuracy on 53 wines the agent never sees:

  | | low effort | medium effort | high effort | cost per run |
  |---|---|---|---|---|
  | **Opus 5.5** | 69.8% | 83.0% | **85.5%** | $1.07 – $1.49 |
  | **Sonnet 5.5** | 49.1% | 55.3% | 65.4% | $0.57 – $0.61 |
  | **Haiku 5.5** | 44.7% | 49.7% | 52.8% | $0.03 – $0.04 |
  | random search (no LLM) | | 57.9% | | $0 |

  These are means of 3 seeds. The start is 38.4%; always predicting the most common class gives 39.6%.
- **What separates the winners is strategy, not effort per se.** Runs that only nudged the random weights did about as well as random search, or worse. All Haiku runs, most Sonnet runs and one Opus run did this. The runs that won treated forward passes as experiments, eight of the nine Opus runs and one Sonnet run among them:
  1. Measure how common each class is.
  2. Rewire the network to test one input at a time.
  3. Hand-build a small linear classifier from the results.

  These runs averaged **82% held-out**, against **52%** for the nudgers.
- **Effort is cheap; model choice is not.** Going from low to high effort adds 8–40% to the cost and about 16 accuracy points for Sonnet and Opus. Opus costs about 40× more per run than Haiku.
- **Leakage matters a lot.** When the prompt included per-class feature averages (experiment 1), every model reached 97–100% within 1–2 passes. A nearest-class-average classifier built from those numbers alone already gets 97.8%.
- **The audits back up the results.** Every transcript was audited for rule-breaking tool calls (no agent read the data, the code or other runs). Every logged pass was re-computed from saved weight snapshots, and all of them match.

## The setup (shared by all experiments)

| | |
|---|---|
| Data | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine): 178 real wines, 13 chemical measurements, 3 cultivars (classes of 59 / 71 / 48). Committed as `data/wine.csv`, with its SHA-256 checked on every load. Features are z-score standardized. |
| Model | MLP `13 → 10 ReLU → 10 ReLU → 3 softmax`: two hidden layers of 10 perceptrons each, 283 parameters, pure numpy (`wsp/model.py`). Weights are stored as readable JSON, one neuron per row. |
| Starting point | Pass 0 is He-normal random weights. Seed *k* gives the same starting weights for every model, effort level and baseline. |
| Budget | 30 forward passes, enforced by the CLI (`python -m wsp forward --run DIR` refuses after 30). An invalid weights file does not use a pass. |
| Agent loop | Read `weights.json` → edit it by hand (Edit/Write tools) → run the forward command → read the output → repeat. The rules allow no other command, no scripts, no gradient or search code, and no files outside the run folder. |
| Agents | Claude Code subagents: Haiku 5.5, Sonnet 5.5 and Opus 5.5, with effort `low` / `medium` / `high` (experiment 1: `low` / `high`), 3 seeds each. |
| Baseline | Random search, a (1+1) evolution strategy. Each pass adds Gaussian noise (σ = 0.1) to every weight and keeps the result if the feedback signal improves. Same budget, same feedback, same seeds; fully deterministic. |

## The three experiments

Each experiment closes the loopholes the previous one exposed.

| | 1: data card | 2: blind | 3: blind + held-out |
|---|---|---|---|
| What the prompt tells the agent about the data | Dataset name, feature names, **per-class feature averages** | Architecture only | Architecture only |
| Inputs and labels | As in the dataset | Shuffled by a fixed permutation | Shuffled by a fixed permutation |
| Feedback after each pass | Accuracy, loss, confusion matrix, dead ReLUs | Accuracy and loss | **Accuracy only**, on 125 feedback wines |
| Scored on | All 178 wines | All 178 wines | **53 held-out wines**, never shown to the agent |
| Prompt | [`agent_prompt.md`](prompts/agent_prompt.md) | [`agent_prompt_blind.md`](prompts/agent_prompt_blind.md) | [`agent_prompt_holdout.md`](prompts/agent_prompt_holdout.md) |
| Runs / results | `runs/`, `results/` | `runs-blind/`, `results-blind/` | `runs-holdout/`, `results-holdout/` |

The input and label shuffling exists because "13 features, 3 classes" might let a model guess the dataset and use what it remembers about it. The permutation (`wsp/data.py`, `BLIND_SEED`) makes such memories useless for setting weights.

### Experiment 3: blind, accuracy-only feedback, scored on held-out wines (main result)

![held-out accuracy per pass](results-holdout/accuracy.png)

**Setup**
- The 178 wines are split 125 / 53 with a fixed, stratified split (`SPLIT_SEED`). Inputs are standardized with feedback-set statistics only.
- Agents see only feedback-set accuracy, which moves in steps of 1/125.
- The score is held-out accuracy of the weights evaluated on pass 30. It is computed afterwards from the saved weight snapshots and is never written anywhere the agent can read.
- Results are in the TL;DR table. Per-trial numbers are in `results-holdout/summary.md`, and every pass is in `results-holdout/all_passes.csv`. `results-holdout/feedback_accuracy.png` shows what the agents saw.

**Findings**
- **Little overfitting.** Feedback accuracy beats held-out accuracy by about 4 to 6 points for Opus, and by about 3 points or less for the others. The classifiers the agents built are simple: one or two weighted sums over a handful of inputs.
- **The probe-then-build pattern is visible in the chart.** Opus's accuracy dips around passes 5–17, while it spends passes on measurements, then jumps once it writes the classifier.

### Experiment 2: blind, accuracy + loss feedback

![blind accuracy per pass](results-blind/accuracy.png)

Accuracy on all 178 wines at pass 30, mean of 3 seeds. The start is 32.2%; always predicting the most common class gives 39.9%.

| | low | medium | high |
|---|---|---|---|
| **Opus** | 74.3% | 84.1% | **86.0%** |
| **Sonnet** | 51.3% | 57.9% | 68.4% |
| **Haiku** | 45.9% | 43.1% | 54.5% |
| random search | 58.4% | | |

The ranking is the same as in experiment 3. Six runs rebuilt the network; they averaged 89.4%, against 55.2% for the 21 runs that nudged weights.

The best run, Opus-high seed 0, reached **97.8% without seeing the data**, but it found a loophole. It sent one input at a time to the outputs at a tiny scale, so that the change in **loss** gave a weighted difference of that input's class averages. From those readings it rebuilt the class averages by hand and wrote a nearest-class-average classifier. That is effectively a hand-computed gradient. It broke no written rule, but it is why experiment 3 drops loss from the feedback.

### Experiment 1: data card in the prompt (leaky)

![experiment 1 accuracy per pass](results/accuracy.png)

| | Haiku | Sonnet | Opus | random search |
|---|---|---|---|---|
| low effort | 98.3% | 98.7% | 99.1% | 58.6% |
| high effort | 98.1% † | 98.5% | 98.5% † | |

These are accuracies on all 178 wines at the last logged pass, means of 3 seeds. † marks configurations that include runs stopped early by the user, ending after 24–26 passes for Haiku and 8 or 15 passes for Opus.

Every agent did the same thing: copy the per-class feature averages from the prompt into three "detector" neurons, pass them straight through, and zero the rest. That alone scores 97.8% (the dashed line); hand-tuning added the last 1–2 points. The experiment measures whether the agent spots the shortcut, not whether it can search weight space.

## What the agents actually did

Counting how many weights each run changed between pass 0 and pass 30 (in `results-*/summary.md`) splits the runs cleanly into two groups.

**Nudgers** (in experiment 3: all Haiku runs, 7 of 9 Sonnet runs and 1 Opus run) left both hidden layers at their random values and edited the output layer and biases: halving, flipping signs, swapping rows, zeroing columns, one change per pass, keeping whatever helped. This is random search done by hand, and it performs like random search or worse.

**Rebuilders** rewrote every hidden weight. They turned the network into an exact linear model by passing chosen inputs through `relu(x)` / `relu(−x)` pairs, with the second layer as a pass-through. Then they used forward passes as experiments:
1. **Class frequencies (2–3 passes).** Zero the weights and set one class's output bias high; the accuracy is then that class's share.
2. **Single-input probes (about 10–15 passes).** Route one input to the outputs ("high → class 0, low → class 2", the middle class as a band near zero). The accuracy shows which way the input points and how strongly.
3. **Build and tune (the remaining passes).** Combine the useful inputs into one or two hand-weighted scores, then adjust the thresholds and biases.

With accuracy-only feedback (experiment 3), 9 of 27 runs rebuilt the network: 8 Opus and 1 Sonnet-high. They averaged 82.2% held-out (73.6–88.7%). The 18 nudgers averaged 51.5%.

This is still learning from the feedback data through the forward pass; there is no other way to learn anything. But every step was a hand-designed experiment with a stated hypothesis, not blind search.

## Cost vs. performance

Cost is the API cost of one 30-pass agent run, computed from the token usage in each transcript. Per-run token counts are committed in `results-*/usage.csv` (extracted with `scripts/extract_usage.py`). `python -m wsp cost` prices them and draws the chart at the top of this README.

Pricing assumptions (`wsp/cost.py`):
- First-party list prices as of November 2026, in USD per million tokens: Haiku 5.5 $0.10 in / $0.50 out, Sonnet 5.5 $2 / $10, Opus 5.5 $4 / $20.
- Cache writes (all 5-minute) are billed at 1.25× the input price.
- Cache reads are $0.20 for Opus and $0.10 for Sonnet. The Haiku cache-read price is assumed to be $0.01, the same 0.1× ratio.
- Every request was served by the requested model, with no fallbacks. No Haiku prompt exceeded 100K tokens, so the standard Haiku rate applies.

Most of the cost is re-reading the growing conversation on each pass, which effort barely changes. That is why raising effort is cheap here. Experiment 2's version of the chart is `results-blind/cost_vs_accuracy.png`.

## Integrity checks

- **Transcript audits.** `scripts/audit_transcript.py` checks every tool call against an allowlist: read the prompt and `weights.json`, edit `weights.json`, run the exact forward command. Results are in `*/trial*/audit.txt` for all 72 agent runs. **No agent read the dataset, the source code, other runs or the held-out scores.** The flags were all harmless slips:
  - no-op shell commands (`cd … && true`, `cp weights.json /dev/null`, an empty `python -`);
  - one bias edit made with `sed` instead of the Edit tool (a hand-chosen value);
  - one forward pass whose output was discarded;
  - edits to mistyped paths that failed;
  - a few documentation lookups or failed message calls;
  - stray empty files in experiment 1, since deleted.
- **Replay.** Every pass's exact weights are saved in `history/pass_XXX.json`. `python -m wsp replay --run DIR` recomputes every logged number, and `report` recomputes all metrics from the snapshots rather than the logs. All runs in all three experiments replay exactly.
- **Experiment 1 deviations.** Four early agents stopped after 2–17 passes and got one nudge to use the full budget. Five runs (Haiku-high ×3, Opus-high ×2) were stopped early by the user and are marked × in the chart. Experiments 2 and 3 had no nudges and no stopped runs; all 54 runs used exactly 30 passes.

## Limitations

- **Small sample.** There are 3 seeds per configuration, and one held-out wine is worth 1.9 points. Differences of a few points between neighbouring configurations are not meaningful; the gaps between models and between strategies are.
- **Small, near-linear dataset.** Wine is close to linearly separable, which favours the build-a-linear-model strategy. A harder dataset might not.
- **Effort setting.** Effort was passed as the subagent effort setting. Whether every model honours every level was not verified independently.
- **Transcripts are not in the repo.** The audit results (`audit.txt`) and token counts (`usage.csv`) are committed, but re-running `audit_transcript.py` or `extract_usage.py` needs the original Claude Code transcripts.
- **One prompt per experiment.** Different wording could change how often models discover the rebuild strategy.

## Reproduce

```bash
pip install -r requirements.txt   # numpy, matplotlib (scikit-learn only for scripts/fetch_data.py)

# verify: recompute every logged pass from its saved weights
python -m wsp replay --run runs-holdout/opus-high/trial0

# rebuild all results, charts and cost plots from the committed runs
python -m wsp report                                                    # experiment 1
python -m wsp report --runs runs-blind   --out results-blind   --models haiku,sonnet,opus --title "Blind protocol (no data info, accuracy+loss feedback only)"
python -m wsp report --runs runs-holdout --out results-holdout --models haiku,sonnet,opus --title "Experiment 3: accuracy-only feedback, scored on held-out set"
python -m wsp cost --results results-blind   --title "Experiment 2: blind, accuracy + loss feedback"
python -m wsp cost --results results-holdout --title "Experiment 3: accuracy-only feedback, scored on held-out set"

# new runs (experiment 3 protocol)
python -m wsp random-search --holdout --run runs-holdout/random-search/trial0 --seed 0   # deterministic baseline
python -m wsp init --holdout --run runs-new/my-model/trial0 --seed 0
python scripts/render_prompt.py runs-new/my-model/trial0 prompts/agent_prompt_holdout.md
#   ...then give runs-new/my-model/trial0/prompt.md to any LLM agent and let it run.
#   Use --blind / prompts/agent_prompt_blind.md for experiment 2, or no flag for experiment 1.
```

## Layout

```
data/wine.csv                 dataset (exported once by scripts/fetch_data.py; checksum-verified)
wsp/data.py                   loading, standardization, blind permutation, held-out split
wsp/model.py                  MLP, random init, forward pass, readable JSON weights
wsp/cli.py                    init / forward (quota) / describe / random-search / replay / report / cost
wsp/report.py                 per-pass metrics from snapshots, summaries, accuracy/loss charts
wsp/cost.py                   token pricing and cost-vs-accuracy chart
prompts/                      the exact prompt templates given to the agents
scripts/render_prompt.py      fills a template for one run (writes RUN/prompt.md)
scripts/audit_transcript.py   checks an agent transcript against the tool allowlist (writes RUN/audit.txt)
scripts/extract_usage.py      per-run token usage from transcripts (writes results-*/usage.csv)
runs*/<config>/trial<k>/      prompt.md, weights.json, log.csv, state.json, audit.txt, history/pass_XXX.json
results*/                     all_passes.csv, summary.md, accuracy.png, loss.png, cost.csv, cost_vs_accuracy.png
```

## Next experiments worth running

- A harder dataset, where a linear model is not enough, to see whether rebuilding still wins.
- Smaller and larger pass budgets, to find where the probe-then-build strategy stops paying off.
- A prompt variant that hints at the probe strategy, to see whether Haiku and Sonnet can carry it out when told.
