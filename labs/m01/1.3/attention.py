"""
Lab 1.3 — Attention, step by step

Single-head causal self-attention on four tokens, printing every intermediate:
Q, K, V projections -> scores -> causal mask -> softmax -> weighted sum of V.

Run:  python labs/m01/1.3/attention.py       (CPU only, needs numpy)
"""

import numpy as np

np.set_printoptions(precision=2, suppress=True)

tokens = ["The", "cat", "sat", "down"]
T, d, dh = len(tokens), 8, 4
rng = np.random.default_rng(1)
x = rng.normal(0, 1, (T, d))                        # one embedding row per token
Wq, Wk, Wv = (rng.normal(0, 0.5, (d, dh)) for _ in range(3))

Q, K, V = x @ Wq, x @ Wk, x @ Wv                     # [T, dh] each
print("Q = x @ Wq:\n", Q, "\n")

S = Q @ K.T / np.sqrt(dh)                            # [T, T]: how much query i likes key j
print("scores S = Q K^T / sqrt(d_head):\n", S, "\n")

mask = np.triu(np.ones((T, T), dtype=bool), k=1)     # True above the diagonal = the future
S_masked = np.where(mask, -np.inf, S)
print("after causal mask (-inf = cannot see the future):\n", S_masked, "\n")

A = np.exp(S_masked - S_masked.max(-1, keepdims=True))
A /= A.sum(-1, keepdims=True)
print("attention weights A = softmax(rows), each row sums to 1:")
for t, row in zip(tokens, A):
    print(f"  {t:>5} attends to", "  ".join(f"{w:.2f}" for w in row))

out = A @ V                                          # [T, dh]
print("\noutput = A @ V:\n", out)

print("\nThe inference-relevant fact: row i only used K[0..i] and V[0..i].")
print("So when a fifth token arrives, rows 0-3 of K and V are unchanged. Cache them (chapter 1.5).")
