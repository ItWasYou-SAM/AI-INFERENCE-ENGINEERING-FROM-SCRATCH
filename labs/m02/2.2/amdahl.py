"""
Lab 2.2 — Parallelism and what "parallelizable" means

1. Amdahl's law: the serial fraction caps your speedup, however many cores you add.
2. Apply it to LLM generation: steps are serial, the math inside a step is not.

Run:  python labs/m02/2.2/amdahl.py           (CPU only, standard library)
"""


def amdahl(p: float, n: int) -> float:
    """Speedup on n workers when a fraction p of the work is parallelizable."""
    return 1 / ((1 - p) + p / n)


if __name__ == "__main__":
    workers = [1, 8, 64, 1024, 16896]          # 16,896 = FP32 lanes on an H100 SXM
    print("Amdahl's law: speedup on N workers\n")
    print(f"{'parallel fraction':>18}" + "".join(f"{n:>9}" for n in workers) + "    limit")
    for p in (0.5, 0.9, 0.99, 0.999, 0.99999):
        print(f"{p:>18}" + "".join(f"{amdahl(p, n):>9.1f}" for n in workers) + f"{1 / (1 - p):>9.0f}")

    # One decode step of Llama-3-8B: ~15 GFLOP of independent multiply-adds.
    print("\nInside one decode step of an 8B model:")
    print("  ~7.5e9 independent multiply-adds per token -> parallel fraction ~ 1.0")
    print("Across decode steps:")
    print("  step t+1 needs the token from step t       -> zero parallelism within one request")
    print("\nSo GPUs parallelize *within* a step (and across requests in a batch),")
    print("never across the steps of a single request.")
