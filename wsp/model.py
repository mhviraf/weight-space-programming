"""A tiny MLP: 13 -> 10 (ReLU) -> 10 (ReLU) -> 3 (softmax).

Two hidden layers of 10 perceptrons each, plus a 3-unit output layer.
Weights live in a human-editable JSON file. W matrices are stored one row
per neuron: layer["W"][j] is the incoming weight vector of neuron j.
"""
import json

import numpy as np

SIZES = [13, 10, 10, 3]
LAYERS = ["hidden1", "hidden2", "output"]


def init_weights(seed: int) -> dict:
    """He-normal random init, rounded to 3 decimals (the random baseline)."""
    rng = np.random.default_rng(seed)
    w = {}
    for name, n_in, n_out in zip(LAYERS, SIZES[:-1], SIZES[1:]):
        W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_out, n_in))
        w[name] = {"W": np.round(W, 3).tolist(), "b": [0.0] * n_out}
    return w


def validate(w: dict) -> None:
    for name, n_in, n_out in zip(LAYERS, SIZES[:-1], SIZES[1:]):
        if name not in w:
            raise ValueError(f"missing layer '{name}'")
        W = np.asarray(w[name]["W"], dtype=float)
        b = np.asarray(w[name]["b"], dtype=float)
        if W.shape != (n_out, n_in):
            raise ValueError(f"{name}.W has shape {W.shape}, expected {(n_out, n_in)}")
        if b.shape != (n_out,):
            raise ValueError(f"{name}.b has shape {b.shape}, expected {(n_out,)}")
        if not (np.isfinite(W).all() and np.isfinite(b).all()):
            raise ValueError(f"{name} contains non-finite values")


def forward(w: dict, X: np.ndarray):
    """Return (probabilities, list of hidden activations)."""
    h = X
    hidden = []
    for name in LAYERS[:-1]:
        h = np.maximum(0.0, h @ np.asarray(w[name]["W"]).T + np.asarray(w[name]["b"]))
        hidden.append(h)
    z = h @ np.asarray(w["output"]["W"]).T + np.asarray(w["output"]["b"])
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True), hidden


def evaluate(w: dict, X: np.ndarray, y: np.ndarray) -> dict:
    p, hidden = forward(w, X)
    pred = p.argmax(axis=1)
    k = p.shape[1]
    conf = np.zeros((k, k), dtype=int)
    for t, q in zip(y, pred):
        conf[t, q] += 1
    return {
        "accuracy": float((pred == y).mean()),
        "loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
        "confusion": conf.tolist(),
        "dead_units": [int((h.max(axis=0) == 0).sum()) for h in hidden],
    }


def dumps(w: dict) -> str:
    """JSON with one neuron (row) per line so it is easy to read and edit."""
    out = ["{"]
    for li, name in enumerate(LAYERS):
        out.append(f'  "{name}": {{')
        out.append('    "W": [')
        rows = w[name]["W"]
        for j, row in enumerate(rows):
            sep = "," if j < len(rows) - 1 else ""
            out.append("      [" + ", ".join(repr(float(v)) for v in row) + "]" + sep)
        out.append("    ],")
        out.append('    "b": [' + ", ".join(repr(float(v)) for v in w[name]["b"]) + "]")
        out.append("  }" + ("," if li < len(LAYERS) - 1 else ""))
    out.append("}")
    return "\n".join(out) + "\n"


def load(path) -> dict:
    with open(path) as f:
        return json.load(f)
