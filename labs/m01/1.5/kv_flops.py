"""
Lab 1.5 — The KV cache: why it exists

Count the work a 7B-class model does to generate token n, with and without
a KV cache, and the memory the cache costs in exchange.

Run:  python labs/m01/1.5/kv_flops.py        (CPU only, standard library)
"""

# Llama-2-7B shape (multi-head attention, no GQA), BF16
L, D, FF, V = 32, 4096, 11008, 32000
BYTES = 2

PARAMS_PER_LAYER = 4 * D * D + 3 * D * FF      # Wq, Wk, Wv, Wo + SwiGLU (gate, up, down)
WEIGHT_FLOPS_PER_TOKEN = 2 * (L * PARAMS_PER_LAYER + D * V)


def flops_no_cache(n: int) -> float:
    """Re-run the whole sequence of n tokens to get token n+1."""
    weights = n * WEIGHT_FLOPS_PER_TOKEN
    attn = L * 2 * (2 * n * n * D)              # QK^T and AV over the full n x n matrix
    return weights + attn


def flops_with_cache(n: int) -> float:
    """Run only the newest token; read n cached K/V rows per layer."""
    weights = WEIGHT_FLOPS_PER_TOKEN
    attn = L * 2 * (2 * n * D)                  # one query row against n keys/values
    return weights + attn


def kv_bytes(n: int) -> float:
    return 2 * L * n * D * BYTES                # K and V, every layer, every position


if __name__ == "__main__":
    print(f"Llama-2-7B: {L} layers, d_model {D}  ->  "
          f"{WEIGHT_FLOPS_PER_TOKEN / 1e9:.1f} GFLOP of weight math per token\n")
    print(f"{'context n':>10}{'no cache':>16}{'KV cache':>14}{'ratio':>9}{'KV memory':>12}")
    for n in (128, 512, 2048, 8192, 32768):
        a, b = flops_no_cache(n), flops_with_cache(n)
        print(f"{n:>10}{a / 1e12:>10.2f} TFLOP{b / 1e9:>9.1f} GFLOP{a / b:>8.0f}x{kv_bytes(n) / 2**30:>9.2f} GiB")

    total_no = sum(flops_no_cache(n) for n in range(1, 2049))
    total_kv = sum(flops_with_cache(n) for n in range(1, 2049))
    print(f"\nGenerating 2,048 tokens from scratch: {total_no / 1e15:.1f} PFLOP without a cache, "
          f"{total_kv / 1e15:.3f} PFLOP with one.")
    print(f"Per-token KV cost: {kv_bytes(1) / 2**20:.2f} MiB. That trade, FLOPs for bytes, "
          "is what the rest of this course manages.")
