"""
Lab 1.1 — Tokens, embeddings and logits

1. Train a tiny byte-pair-encoding (BPE) tokenizer on a paragraph and watch merges form.
2. Look up embeddings for the tokens (a table lookup, not a matmul).
3. Turn a final hidden state into logits and probabilities over the vocabulary.
4. Size the embedding and unembedding tables of real models.

Run:  python labs/m01/1.1/tokens.py          (CPU only, needs numpy)
"""

from collections import Counter

import numpy as np

TEXT = (
    "the cache stores the keys and the values so the model never recomputes them. "
    "the scheduler batches the requests so the gpu never waits for them."
)


def train_bpe(text: str, n_merges: int):
    ids = list(text.encode("utf-8"))                 # start from raw bytes: vocab = 256
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for _ in range(n_merges):
        pairs = Counter(zip(ids, ids[1:]))
        (a, b), count = pairs.most_common(1)[0]
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((a, b, new_id, count))
        out, i = [], 0
        while i < len(ids):                          # replace every (a, b) with new_id
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids = out
    return ids, vocab, merges


if __name__ == "__main__":
    n_bytes = len(TEXT.encode())
    ids, vocab, merges = train_bpe(TEXT, n_merges=20)
    print(f"text: {n_bytes} bytes")
    print("first merges (most frequent pair becomes a new token):")
    for a, b, new, count in merges[:8]:
        print(f"  {vocab[a]!r:>8} + {vocab[b]!r:<8} -> id {new} {vocab[new]!r}  (seen {count}x)")
    print(f"after 20 merges: {len(ids)} tokens for {n_bytes} bytes "
          f"({n_bytes / len(ids):.2f} bytes/token)")
    print("tokenized:", " | ".join(vocab[i].decode(errors='replace') for i in ids[:14]), "| ...\n")

    # embeddings: one row per token id
    rng = np.random.default_rng(0)
    V, d = len(vocab), 8
    E = rng.normal(0, 1, (V, d)).astype(np.float32)
    x = E[ids[:5]]                                    # gather rows: shape [5, d]
    print(f"embedding table E: {E.shape}, first 5 tokens -> x: {x.shape}")

    # logits: last hidden state times the (tied) unembedding
    h = x[-1]                                          # pretend this left the last layer
    logits = E @ h                                     # [V]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    seen = sorted(set(ids))                            # only show tokens that occur in the text
    top = sorted(seen, key=lambda i: -p[i])[:3]
    print("top-3 next tokens (random weights, so meaningless):",
          [(vocab[i].decode(), round(float(p[i]), 3)) for i in top])

    print("\nReal models: embedding table = vocab x d_model x 2 bytes (BF16)")
    for name, vocab_size, d_model in [("Llama-2-7B", 32000, 4096),
                                      ("Llama-3-8B", 128256, 4096),
                                      ("Qwen2.5-7B", 152064, 3584)]:
        mb = vocab_size * d_model * 2 / 1e6
        print(f"  {name:<11} {vocab_size:>7} x {d_model} -> {mb:6.0f} MB "
              f"(and the same again for the LM head if untied)")
