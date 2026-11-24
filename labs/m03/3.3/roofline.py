"""
Lab 3.3 — The roofline model

For one d x d FP16 linear layer (Y = X @ W, X is M x d), compute arithmetic
intensity and the attainable throughput on a few GPUs. M is the number of tokens
processed in one step: the batch size during decode, the prompt length during prefill.

Run:  python labs/m03/3.3/roofline.py      (CPU only, standard library)
"""

D = 4096
GPUS = {  # dense BF16/FP16 tensor TFLOP/s, memory TB/s (datasheet numbers)
    "L4": (121, 0.30),
    "RTX 4090": (165, 1.01),
    "A100 80GB": (312, 2.04),
    "H100 SXM": (989, 3.35),
}


def flops(m: int, d: int = D) -> float:
    return 2 * m * d * d  # one multiply + one add per weight per token


def bytes_moved(m: int, d: int = D) -> float:
    return 2 * (m * d + d * d + m * d)  # read X, read W, write Y (FP16 = 2 B)


def intensity(m: int) -> float:
    return flops(m) / bytes_moved(m)


def attainable(gpu: str, ai: float) -> float:
    peak, bw = GPUS[gpu]
    return min(peak, bw * ai)  # TB/s * FLOP/B = TFLOP/s


def ridge_batch(gpu: str, d: int = D) -> float:
    """Smallest M where the layer stops being memory-bound: solve M*d/(d+2M) = ridge."""
    peak, bw = GPUS[gpu]
    r = peak / bw
    return r * d / (d - 2 * r)


if __name__ == "__main__":
    gpu = "H100 SXM"
    peak, bw = GPUS[gpu]
    print(f"{gpu}: peak {peak} TFLOP/s, {bw} TB/s -> ridge {peak / bw:.0f} FLOP/B\n")
    print(f"{'phase':<8}{'M':>7}{'AI (FLOP/B)':>14}{'TFLOP/s':>10}{'% peak':>9}  bound")
    for phase, m in [("decode", 1), ("decode", 8), ("decode", 64), ("decode", 256),
                     ("prefill", 512), ("prefill", 2048)]:
        ai = intensity(m)
        perf = attainable(gpu, ai)
        bound = "memory" if ai < peak / bw else "compute"
        print(f"{phase:<8}{m:>7}{ai:>14.1f}{perf:>10.1f}{perf / peak * 100:>8.1f}%  {bound}")

    print("\nDecode batch size needed to reach the ridge (d = 4096):")
    for g in GPUS:
        print(f"  {g:<10} ~{ridge_batch(g):.0f} sequences")
