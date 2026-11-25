"""
Lab 1.4 — Autoregressive generation and sampling basics

Given one set of logits, show how greedy decoding and temperature change the
next-token distribution, then sample from each to see the variety.

Run:  python labs/m01/1.4/sampling.py        (CPU only, needs numpy)
"""

import numpy as np

vocab = ["Paris", "the", "a", "France", "Lyon", "beautiful", "known", "located"]
logits = np.array([6.0, 4.2, 3.9, 3.1, 2.5, 2.2, 1.8, 1.0])   # "The capital of France is ___"


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample(logits, temperature, rng):
    if temperature == 0:
        return int(np.argmax(logits))                # greedy: always the top token
    p = softmax(logits / temperature)
    return int(rng.choice(len(p), p=p))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'token':<10}" + "".join(f"T={t:<6}" for t in (0.5, 1.0, 1.5)))
    probs = {t: softmax(logits / t) for t in (0.5, 1.0, 1.5)}
    for i, tok in enumerate(vocab):
        print(f"{tok:<10}" + "".join(f"{probs[t][i]:<8.3f}" for t in (0.5, 1.0, 1.5)))

    print("\n20 samples at each temperature:")
    for t in (0, 0.5, 1.0, 1.5):
        picks = [vocab[sample(logits, t, rng)] for _ in range(20)]
        distinct = len(set(picks))
        print(f"  T={t:<4} {distinct} distinct  e.g. {', '.join(picks[:6])}")

    print("\nCost note: sampling is microseconds. The forward pass that produced the")
    print("logits is milliseconds. Generation is slow because of the loop, not the dice.")
