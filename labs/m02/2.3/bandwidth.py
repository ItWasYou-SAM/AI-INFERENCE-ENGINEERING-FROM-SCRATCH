"""
Lab 2.3 — Memory bandwidth and the memory hierarchy

Measure how fast your own machine streams data as the working set grows.
Small arrays live in CPU caches; large ones come from DRAM. Watch the cliff.

Run:  python labs/m02/2.3/bandwidth.py        (CPU only, needs numpy)
"""

import time

import numpy as np


def stream_gbps(n_bytes: int, min_time: float = 0.2) -> float:
    """Best-of-N read+write bandwidth of dst[:] = src for an array of n_bytes."""
    n = max(1, n_bytes // 8)
    src = np.ones(n, dtype=np.float64)
    dst = np.empty_like(src)
    np.copyto(dst, src)                                 # warm up: touch every page
    reps = max(1, int(2e8 // n_bytes))                  # enough reps to be measurable
    best = float("inf")
    deadline = time.perf_counter() + min_time
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            np.copyto(dst, src)
        best = min(best, (time.perf_counter() - t0) / reps)
        if time.perf_counter() > deadline:
            break
    return 2 * n * 8 / best / 1e9                       # read src + write dst


if __name__ == "__main__":
    print(f"{'working set':>12}  {'GB/s':>7}")
    sizes = [16 * 2**10, 128 * 2**10, 1 * 2**20, 8 * 2**20, 32 * 2**20, 256 * 2**20, 1 * 2**30]
    results = []
    for size in sizes:
        gbps = stream_gbps(size)
        results.append(gbps)
        label = f"{size // 2**20} MiB" if size >= 2**20 else f"{size // 2**10} KiB"
        print(f"{label:>12}  {gbps:7.1f}  {'#' * int(gbps / 4)}")

    dram = results[-1]
    print("\n(One thread doing a plain copy. All cores together get more, but not 100x more.)")
    print(f"DRAM bandwidth here: ~{dram:.0f} GB/s.  An H100's HBM: ~3,350 GB/s "
          f"(~{3350 / dram:.0f}x).")
    print(f"Time to read 16 GB of 8B-model weights once: this machine ~{16 / dram:.2f} s, "
          f"H100 ~{16 / 3350 * 1000:.1f} ms.")
    print("That read happens on every decode step. Bandwidth is the speed limit for decode.")
