You are doing a "weight-space programming" experiment. You will try to improve
a small neural network purely by editing its weights by hand and observing the
result of forward passes. You must NOT train it.

## The network
A classifier: 13 inputs -> 10 ReLU -> 10 ReLU -> 3 logits -> softmax.
The weights are in `/home/user/weight-space-programming-/runs-blind/opus-medium/trial2/weights.json`, with three layers: `hidden1`
(W: 10 rows x 13), `hidden2` (W: 10 rows x 10), `output` (W: 3 rows x 10), each
with a bias vector `b`. Row j of W is the incoming weight vector of neuron j,
so neuron j computes relu(W[j] . x + b[j]) (no relu on the output layer).
The weights start random (pass 0 = random baseline).

You know nothing about the data or what the inputs and classes mean, and you
must not try to find out. All you have is the weights and the output of each
forward pass.

## Your loop
You have a hard quota of exactly 30 forward passes. Each pass evaluates the
current `/home/user/weight-space-programming-/runs-blind/opus-medium/trial2/weights.json` and prints ONLY the accuracy and the mean
cross-entropy loss. Run it with:

    cd /home/user/weight-space-programming- && python -m wsp forward --run /home/user/weight-space-programming-/runs-blind/opus-medium/trial2

Repeat: look at the weights and the latest output -> decide on a change ->
edit weights.json -> run one forward pass -> read the result.
You MUST use all 30 passes; do not stop early, even if results look good.
If you make the file invalid, the command tells you and no pass is consumed.

## Rules (strict)
- The ONLY command you may run is the forward command above, exactly as written.
  Do not run python, scripts, calculators, or any other shell command.
- Do not compute gradients, and do not write or run any optimizer, search or
  fitting code. Every weight change must come from your own reasoning, typed by hand.
- Only read/edit files inside `/home/user/weight-space-programming-/runs-blind/opus-medium/trial2/`: Read `weights.json`, and edit it
  with the Edit or Write tool. Do not read anything else in the repository
  (no dataset, no source code, no other runs, no other files).
- Do not create any other files.
- The log keeps the result of every pass, not the best one, so the
  weights you leave on the last pass are what counts at the end.

When done, reply with a short summary (under 150 words): your strategy, final
accuracy, and confirm you followed the rules (list the kinds of commands/tools
you used).
