You are doing a "weight-space programming" experiment. You will improve a tiny
neural network by hand-editing its weights. You must NOT train it.

## The model
An MLP classifier for the UCI Wine dataset (178 real wines, 13 chemical
features, 3 cultivars): 13 inputs -> 10 ReLU -> 10 ReLU -> 3 logits -> softmax.
The weights are in `{RUN}/weights.json`, which has three layers: `hidden1`
(W: 10 rows x 13), `hidden2` (W: 10 rows x 10), `output` (W: 3 rows x 10), each
with a bias vector `b`. Row j of W is the incoming weight vector of neuron j,
so neuron j computes relu(W[j] . x + b[j]) (no relu on the output layer).
The weights start random (pass 0 = random baseline).

## The data card (inputs are z-score standardized; this is all you get about the data)
{DATA_CARD}

## Your loop
You have a hard quota of 30 forward passes. Each pass evaluates the current
`{RUN}/weights.json` on the entire dataset and is logged automatically:

    cd {REPO} && python -m wsp forward --run {RUN}

It prints accuracy, cross-entropy loss, the confusion matrix and how many ReLUs
are dead. Repeat: think -> edit weights.json -> run one forward pass -> read the
result. Use all 30 passes (the quota is enforced; after 30 the command refuses).
If you make the file invalid, the command tells you and no pass is consumed.

## Rules (strict)
- The ONLY command you may run is the forward command above, exactly as written.
  Do not run python, scripts, calculators, or any other shell command.
- Do not compute gradients or write/run any optimizer, search or fitting code.
  Every weight change must come from your own reasoning, typed by hand.
- Only read/edit files inside `{RUN}/` (you may Read `weights.json`, and edit it
  with the Edit or Write tool). Do not read the dataset file, other runs, the
  `wsp/` source, or anything else in the repo.
- Note: the log keeps the result of every pass, not the best one, so the
  weights you leave on the last pass are what counts at the end.

When done, reply with a short summary (under 150 words): your strategy, final
accuracy, and confirm you followed the rules (list the kinds of commands/tools
you used).
