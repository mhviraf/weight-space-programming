You are doing a "weight-space programming" experiment. You will improve a tiny
neural network by hand-editing its weights. You must NOT train it.

## The model
An MLP classifier for the UCI Wine dataset (178 real wines, 13 chemical
features, 3 cultivars): 13 inputs -> 10 ReLU -> 10 ReLU -> 3 logits -> softmax.
The weights are in `/home/user/weight-space-programming-/runs/opus-high/trial0/weights.json`, which has three layers: `hidden1`
(W: 10 rows x 13), `hidden2` (W: 10 rows x 10), `output` (W: 3 rows x 10), each
with a bias vector `b`. Row j of W is the incoming weight vector of neuron j,
so neuron j computes relu(W[j] . x + b[j]) (no relu on the output layer).
The weights start random (pass 0 = random baseline).

## The data card (inputs are z-score standardized; this is all you get about the data)
Wine dataset: 178 samples, 13 features, 3 classes (counts: [59, 71, 48])
Features are z-score standardized (mean 0, std 1 over the whole dataset).
Per-class mean of each standardized feature (input index: name):
  idx feature                          class 0   class 1   class 2
    0 alcohol                             0.92     -0.89      0.19
    1 malic_acid                         -0.29     -0.36      0.90
    2 ash                                 0.33     -0.44      0.26
    3 alcalinity_of_ash                  -0.74      0.22      0.58
    4 magnesium                           0.46     -0.36     -0.03
    5 total_phenols                       0.87     -0.06     -0.99
    6 flavanoids                          0.96      0.05     -1.25
    7 nonflavanoid_phenols               -0.58      0.01      0.69
    8 proanthocyanins                     0.54      0.07     -0.77
    9 color_intensity                     0.20     -0.85      1.01
   10 hue                                 0.46      0.43     -1.21
   11 od280/od315_of_diluted_wines        0.77      0.25     -1.31
   12 proline                             1.17     -0.72     -0.37


## Your loop
You have a hard quota of 30 forward passes. Each pass evaluates the current
`/home/user/weight-space-programming-/runs/opus-high/trial0/weights.json` on the entire dataset and is logged automatically:

    cd /home/user/weight-space-programming- && python -m wsp forward --run /home/user/weight-space-programming-/runs/opus-high/trial0

It prints accuracy, cross-entropy loss, the confusion matrix and how many ReLUs
are dead. Repeat: think -> edit weights.json -> run one forward pass -> read the
result. Use all 30 passes (the quota is enforced; after 30 the command refuses).
If you make the file invalid, the command tells you and no pass is consumed.

## Rules (strict)
- The ONLY command you may run is the forward command above, exactly as written.
  Do not run python, scripts, calculators, or any other shell command.
- Do not compute gradients or write/run any optimizer, search or fitting code.
  Every weight change must come from your own reasoning, typed by hand.
- Only read/edit files inside `/home/user/weight-space-programming-/runs/opus-high/trial0/` (you may Read `weights.json`, and edit it
  with the Edit or Write tool). Do not read the dataset file, other runs, the
  `wsp/` source, or anything else in the repo.
- Note: the log keeps the result of every pass, not the best one, so the
  weights you leave on the last pass are what counts at the end.

When done, reply with a short summary (under 150 words): your strategy, final
accuracy, and confirm you followed the rules (list the kinds of commands/tools
you used).
