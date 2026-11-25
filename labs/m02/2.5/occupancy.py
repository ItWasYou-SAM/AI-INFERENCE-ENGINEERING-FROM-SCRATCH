"""
Lab 2.5 — Occupancy and latency hiding

1. Occupancy calculator: how many warps fit on one SM, and which resource runs out first.
2. Latency hiding: how many warps a scheduler needs to stay busy while loads are in flight.

Limits are for an H100 (compute capability 9.0). Real allocation has rounding
granularity that this simple version ignores.

Run:  python labs/m02/2.5/occupancy.py        (CPU only, standard library)
"""

SM = dict(max_warps=64, max_blocks=32, regs=65536, smem=228 * 1024, schedulers=4)


def occupancy(threads_per_block: int, regs_per_thread: int, smem_per_block: int):
    warps_per_block = -(-threads_per_block // 32)
    limits = {
        "warps": SM["max_warps"] // warps_per_block,
        "blocks": SM["max_blocks"],
        "registers": SM["regs"] // (regs_per_thread * threads_per_block),
        "shared mem": SM["smem"] // smem_per_block if smem_per_block else SM["max_blocks"],
    }
    blocks = min(limits.values())
    limiter = min(limits, key=limits.get)
    warps = blocks * warps_per_block
    return warps, warps / SM["max_warps"], limiter


def utilization(warps_per_scheduler: float, compute: int, latency: int) -> float:
    """Each warp issues `compute` cycles of work, then waits `latency` cycles on memory."""
    return min(1.0, warps_per_scheduler * compute / (compute + latency))


if __name__ == "__main__":
    print("Occupancy (H100 SM: 64 warps, 64K registers, 228 KB shared memory)\n")
    print(f"{'threads/blk':>11}{'regs/thr':>9}{'smem/blk':>10}{'warps':>7}{'occ.':>7}  limited by")
    for tpb, regs, smem in [(256, 32, 0), (256, 64, 0), (256, 128, 0), (256, 255, 0),
                            (128, 64, 48 * 1024), (256, 64, 100 * 1024)]:
        warps, occ, lim = occupancy(tpb, regs, smem)
        print(f"{tpb:>11}{regs:>9}{smem // 1024:>8}KB{warps:>7}{occ:>7.0%}  {lim}")

    print("\nLatency hiding: 20 cycles of math per load, ~500-cycle memory latency")
    for w in (1, 2, 4, 8, 16):
        print(f"  {w:>2} warps/scheduler -> scheduler busy {utilization(w, 20, 500):6.1%}")
    need = (20 + 500) / 20
    print(f"  needed for 100%: ~{need:.0f} warps/scheduler, but an SM holds at most 16 per scheduler.")
    print("  So kernels also hide latency with more bytes in flight per warp (ILP, vector loads, async copies).")
