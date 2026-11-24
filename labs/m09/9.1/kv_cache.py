"""
Lab 9.1 — How big is the KV cache?

Computes KV-cache bytes per token, per user, and the maximum number of concurrent
users a GPU can hold (the capacity equation). Then checks the per-token formula
against an actual NumPy KV cache.

Run:  python labs/m09/9.1/kv_cache.py          (CPU only, needs numpy)
Units: decimal GB (1 GB = 1e9 bytes).
"""
from dataclasses import dataclass

import numpy as np

GB = 1e9


@dataclass
class Model:
    name: str
    params_b: float  # parameters, in billions
    layers: int
    kv_heads: int
    head_dim: int


MISTRAL_7B = Model("Mistral 7B", 7.24, layers=32, kv_heads=8, head_dim=128)
LLAMA2_7B = Model("Llama 2 7B (MHA)", 6.74, layers=32, kv_heads=32, head_dim=128)
LLAMA31_70B = Model("Llama 3.1 70B", 70.6, layers=80, kv_heads=8, head_dim=128)


def kv_bytes_per_token(m: Model, kv_bytes: float = 2) -> float:
    # 2 tensors (K and V) per layer, each kv_heads x head_dim values
    return 2 * m.layers * m.kv_heads * m.head_dim * kv_bytes


def max_users(m: Model, gpu_gb: float, context: int, weight_bytes: float = 2,
              kv_bytes: float = 2, overhead_gb: float = 8) -> int:
    weights = m.params_b * 1e9 * weight_bytes
    free = gpu_gb * GB - weights - overhead_gb * GB
    if free <= 0:
        return 0
    return int(free // (kv_bytes_per_token(m, kv_bytes) * context))


def measure_with_numpy(m: Model, tokens: int) -> int:
    """Allocate a real FP16 KV cache (one sequence) and count its bytes."""
    cache = [
        (np.zeros((m.kv_heads, tokens, m.head_dim), dtype=np.float16),   # K
         np.zeros((m.kv_heads, tokens, m.head_dim), dtype=np.float16))   # V
        for _ in range(m.layers)
    ]
    return sum(k.nbytes + v.nbytes for k, v in cache)


if __name__ == "__main__":
    m = MISTRAL_7B
    per_tok = kv_bytes_per_token(m)
    print(f"{m.name}: KV per token = {per_tok:,.0f} B ≈ {per_tok / 1e3:.0f} KB")

    measured = measure_with_numpy(m, tokens=256)
    assert measured == per_tok * 256, "formula and real allocation disagree"
    print(f"NumPy check: 256 tokens -> {measured:,} B (matches formula)\n")

    print("Per-user KV at different context lengths:")
    for ctx in [1024, 4096, 16384, 32768, 131072]:
        print(f"  {ctx:>7,} tokens -> {per_tok * ctx / GB:6.2f} GB")

    print("\nCapacity on an 80 GB GPU (FP16 weights + KV, 8 GB overhead):")
    for ctx in [4096, 16384]:
        print(f"  {m.name} @ {ctx // 1024}K -> {max_users(m, 80, ctx)} users")

    print("\nExperiments:")
    print(f"  Llama 2 7B (no GQA) @ 4K           -> {max_users(LLAMA2_7B, 80, 4096)} users")
    print(f"  Mistral 7B, FP8 KV @ 4K            -> {max_users(m, 80, 4096, kv_bytes=1)} users")
    print(f"  Llama 3.1 70B FP16 on 80 GB        -> {max_users(LLAMA31_70B, 80, 4096)} users (weights alone: {70.6 * 2:.0f} GB)")
    print(f"  Llama 3.1 70B INT4 weights @ 4K    -> {max_users(LLAMA31_70B, 80, 4096, weight_bytes=0.5)} users")
