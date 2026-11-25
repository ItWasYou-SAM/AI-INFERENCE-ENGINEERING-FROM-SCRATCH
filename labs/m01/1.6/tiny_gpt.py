"""
Lab 1.6 — A tiny GPT inference loop in NumPy

A complete decoder-only transformer (pre-norm, RMSNorm, causal multi-head
attention, GELU MLP, tied embeddings) running autoregressive generation two ways:

  1. naive:    recompute the whole sequence every step   (O(n^2) work per token)
  2. kv-cache: prefill once, then one token per step     (O(n)   work per token)

The weights are random, so the text is gibberish. What matters is the mechanics:
both loops must produce the *same tokens*, and the cached loop must be faster.

Run:  python labs/m01/1.6/tiny_gpt.py          (CPU only, needs numpy)
"""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    vocab: int = 256      # byte-level vocabulary: token id = byte value
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq: int = 512


# ---------------------------------------------------------------- weights

def init_weights(cfg: Config, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    std = 0.02
    w = {
        "tok_emb": rng.normal(0, std, (cfg.vocab, cfg.d_model)).astype(np.float32),
        "pos_emb": rng.normal(0, std, (cfg.max_seq, cfg.d_model)).astype(np.float32),
        "norm_f": np.ones(cfg.d_model, np.float32),
        "layers": [],
    }
    for _ in range(cfg.n_layers):
        w["layers"].append({
            "norm1": np.ones(cfg.d_model, np.float32),
            "w_qkv": rng.normal(0, std, (cfg.d_model, 3 * cfg.d_model)).astype(np.float32),
            "w_o": rng.normal(0, std, (cfg.d_model, cfg.d_model)).astype(np.float32),
            "norm2": np.ones(cfg.d_model, np.float32),
            "w_up": rng.normal(0, std, (cfg.d_model, cfg.d_ff)).astype(np.float32),
            "w_down": rng.normal(0, std, (cfg.d_ff, cfg.d_model)).astype(np.float32),
        })
    return w


def count_params(w: dict) -> int:
    n = w["tok_emb"].size + w["pos_emb"].size + w["norm_f"].size
    for layer in w["layers"]:
        n += sum(v.size for v in layer.values())
    return n


# ---------------------------------------------------------------- building blocks

def rms_norm(x, g, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)      # subtract max for numerical safety
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def split_heads(x, n_heads):                      # [T, d] -> [H, T, d_head]
    T, d = x.shape
    return x.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)


def merge_heads(x):                               # [H, T, d_head] -> [T, d]
    H, T, dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, H * dh)


def attention(q, k, v, q_offset):
    """q: [H, Tq, dh]; k, v: [H, Tk, dh]. Query i sits at absolute position q_offset + i."""
    dh = q.shape[-1]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dh)          # [H, Tq, Tk]
    Tq, Tk = scores.shape[1:]
    q_pos = q_offset + np.arange(Tq)[:, None]
    k_pos = np.arange(Tk)[None, :]
    scores = np.where(k_pos <= q_pos, scores, -np.inf)       # causal mask: no peeking ahead
    return softmax(scores) @ v                               # [H, Tq, dh]


# ---------------------------------------------------------------- forward pass

def forward(w, cfg, tokens, start_pos=0, cache=None):
    """
    Run the model on `tokens` (positions start_pos .. start_pos+len-1).
    If `cache` is a list of per-layer {"k", "v"} dicts, new K/V are appended to it
    and attention reads the whole cache. Returns logits for every input position.
    """
    T = len(tokens)
    x = w["tok_emb"][tokens] + w["pos_emb"][start_pos:start_pos + T]   # [T, d]

    for i, layer in enumerate(w["layers"]):
        h = rms_norm(x, layer["norm1"])
        q, k, v = np.split(h @ layer["w_qkv"], 3, axis=-1)
        q, k, v = (split_heads(t, cfg.n_heads) for t in (q, k, v))
        if cache is not None:
            c = cache[i]
            c["k"] = k if c["k"] is None else np.concatenate([c["k"], k], axis=1)
            c["v"] = v if c["v"] is None else np.concatenate([c["v"], v], axis=1)
            k, v = c["k"], c["v"]
        x = x + merge_heads(attention(q, k, v, start_pos)) @ layer["w_o"]
        x = x + gelu(rms_norm(x, layer["norm2"]) @ layer["w_up"]) @ layer["w_down"]

    return rms_norm(x, w["norm_f"]) @ w["tok_emb"].T        # tied unembedding: [T, vocab]


# ---------------------------------------------------------------- generation loops

def generate_naive(w, cfg, prompt, n_new):
    tokens = list(prompt)
    for _ in range(n_new):
        logits = forward(w, cfg, np.array(tokens))           # whole sequence, every step
        tokens.append(int(logits[-1].argmax()))
    return tokens[len(prompt):]


def generate_cached(w, cfg, prompt, n_new):
    cache = [{"k": None, "v": None} for _ in range(cfg.n_layers)]
    logits = forward(w, cfg, np.array(prompt), 0, cache)     # prefill: all prompt tokens at once
    out = [int(logits[-1].argmax())]
    for _ in range(n_new - 1):                               # decode: one token per step
        pos = len(prompt) + len(out) - 1
        logits = forward(w, cfg, np.array([out[-1]]), pos, cache)
        out.append(int(logits[-1].argmax()))
    kv_bytes = sum(c["k"].nbytes + c["v"].nbytes for c in cache)
    return out, kv_bytes


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


if __name__ == "__main__":
    cfg = Config()
    w = init_weights(cfg)
    prompt = list(b"The KV cache exists because")
    n_new = 100

    print(f"tiny GPT: {cfg.n_layers} layers, d_model {cfg.d_model}, {cfg.n_heads} heads, "
          f"{count_params(w) / 1e6:.2f}M params")
    print(f"prompt: {len(prompt)} tokens, generating {n_new} greedy tokens\n")

    naive, t_naive = timed(generate_naive, w, cfg, prompt, n_new)
    (cached, kv_bytes), t_cached = timed(generate_cached, w, cfg, prompt, n_new)

    print(f"naive    (recompute everything): {t_naive * 1000:8.1f} ms")
    print(f"kv-cache (prefill + decode)    : {t_cached * 1000:8.1f} ms   "
          f"-> {t_naive / t_cached:.1f}x faster")
    print(f"same tokens: {naive == cached}")
    print(f"KV cache at the end: {kv_bytes / 1024:.0f} KiB "
          f"= 2 (K,V) x {cfg.n_layers} layers x {len(prompt) + n_new - 1} positions "
          f"x {cfg.d_model} dims x 4 bytes")
    print("first 16 generated ids:", cached[:16])
