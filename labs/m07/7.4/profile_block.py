"""Lab 7.4 — The PyTorch profiler and traces

A small GPT-style stack (6 layers, d = 1024, FP16) profiled with torch.profiler in
three situations:

  prefill         1 sequence of 512 tokens: big matmuls, the GPU should stay busy
  prefill + sync  the same, plus a "debug" line in every layer that calls .item()
  decode          1 new token at the end of a 512-token context: the same layers,
                  tiny matmuls, many small kernels

For each one the lab prints the wall-clock step time, how much of the step the GPU
spent running kernels (from the exported trace), the CPU time spent launching, and
the profiler's table of the kernels that took the most GPU time.
Traces are written to traces/*.json; open them at https://ui.perfetto.dev.

Run:  python profile_block.py            (on a CUDA GPU; on CPU it runs, with CPU-only tables)
      nsys profile -t cuda,nvtx --pytorch=autograd-nvtx python profile_block.py --nsys
            --nsys skips torch.profiler (the two can't both drive the autograd profiler)
            and marks each step with an NVTX range instead
"""
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function, schedule

DEV = "cuda" if torch.cuda.is_available() else "cpu"
UNDER_NSYS = "--nsys" in sys.argv
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
D, HEADS, LAYERS, CTX = 1024, 16, 6, 512


class Block(nn.Module):
    def __init__(self, debug_sync=False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.up, self.down = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        self.debug_sync = debug_sync

    def forward(self, x, k_cache, v_cache, pos):
        B, T, _ = x.shape
        with record_function("attention"):
            q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
            q, k, v = (t.view(B, T, HEADS, D // HEADS).transpose(1, 2) for t in (q, k, v))
            k_cache[:, :, pos : pos + T], v_cache[:, :, pos : pos + T] = k, v
            k, v = k_cache[:, :, : pos + T], v_cache[:, :, : pos + T]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
            x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        with record_function("mlp"):
            x = x + self.down(F.gelu(self.up(self.ln2(x))))
        if self.debug_sync and x.abs().max().item() > 1e4:   # a harmless-looking check
            print("activation blew up")
        return x


class Model(nn.Module):
    def __init__(self, debug_sync=False):
        super().__init__()
        self.blocks = nn.ModuleList(Block(debug_sync) for _ in range(LAYERS))

    def forward(self, x, caches, pos):
        for blk, (kc, vc) in zip(self.blocks, caches):
            x = blk(x, kc, vc, pos)
        return x


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def gpu_busy_fraction(trace_path):
    """Share of the profiled span during which at least one kernel was running."""
    with open(trace_path) as f:
        ev = json.load(f)["traceEvents"]
    kernels = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "kernel" and "dur" in e)
    steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep#") and "dur" in e]
    if not kernels or not steps:
        return None
    t0, t1 = min(e["ts"] for e in steps), max(e["ts"] + e["dur"] for e in steps)
    busy, end = 0.0, t0
    for s, e in kernels:                       # union of kernel intervals inside [t0, t1]
        s, e = max(s, end, t0), min(e, t1)
        if e > s:
            busy, end = busy + (e - s), e
    return busy / (t1 - t0), len(kernels)


@torch.no_grad()
def run(name, tokens, debug_sync=False):
    torch.manual_seed(0)
    model = Model(debug_sync).to(DEV, DTYPE).eval()
    shape = (1, HEADS, CTX + 8, D // HEADS)
    caches = [(torch.zeros(shape, device=DEV, dtype=DTYPE), torch.zeros(shape, device=DEV, dtype=DTYPE)) for _ in range(LAYERS)]
    x = torch.randn(1, tokens, D, device=DEV, dtype=DTYPE)
    pos = 0 if tokens > 1 else CTX - 1         # decode: one token at the end of a 512-token context

    for _ in range(5):                         # warm-up outside the profiler
        model(x, caches, pos)
    sync()
    windows = []                               # 7 windows of 20 steps: median and spread
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(20):
            model(x, caches, pos)
        sync()
        windows.append((time.perf_counter() - t) / 20 * 1e3)
    windows.sort()
    wall_ms = windows[3]

    flop = 2 * tokens * LAYERS * (3 * D * D + D * D + 8 * D * D)
    print(f"\n=== {name}: {tokens} token(s), {LAYERS} layers, d = {D}, {DTYPE}  ({DEV})")
    print(f"step (wall clock, no profiler): {wall_ms:.2f} ms (range {windows[0]:.2f}-{windows[-1]:.2f})"
          f"   matmul TFLOP/s: {flop / wall_ms / 1e9:.2f}")
    if UNDER_NSYS:                             # a few named steps for the nsys timeline
        for i in range(5):
            torch.cuda.nvtx.range_push(f"{name} step {i}")
            model(x, caches, pos)
            sync()
            torch.cuda.nvtx.range_pop()
        return

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEV == "cuda" else [])
    with profile(activities=acts, schedule=schedule(wait=1, warmup=2, active=5)) as prof:
        for _ in range(8):
            model(x, caches, pos)
            sync()
            prof.step()

    os.makedirs("traces", exist_ok=True)
    path = f"traces/{name.replace(' ', '_').replace('+', 'plus')}.json"
    prof.export_chrome_trace(path)
    busy = gpu_busy_fraction(path) if DEV == "cuda" else None
    if busy:
        print(f"GPU busy: {busy[0]:.0%} of the profiled steps ({busy[1] // 5} kernels per step)")
    steps = 5
    avgs = prof.key_averages()
    launch = [e for e in avgs if "LaunchKernel" in e.key]
    if launch:
        n = sum(e.count for e in launch) / steps
        us = sum(e.self_cpu_time_total for e in launch) / steps
        print(f"CPU in kernel launches: {n:.0f} launches, {us / 1e3:.2f} ms per step (with profiler overhead)")
    syncs = [e for e in avgs if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize")]
    for e in syncs:
        print(f"CPU waiting in {e.key}: {e.count / steps:.0f} calls, {e.self_cpu_time_total / steps / 1e3:.2f} ms per step")
    if DEV == "cuda":
        # Kernels have GPU time but no CPU time of their own; leave out the record_function labels.
        names = {"attention", "mlp"}
        kern = [e for e in avgs if e.self_device_time_total > 0 and e.self_cpu_time_total == 0
                and e.key not in names and not e.key.startswith("ProfilerStep")]
        kern.sort(key=lambda e: -e.self_device_time_total)
        total = sum(e.self_device_time_total for e in kern)
        print(f"kernel time: {total / steps / 1e3:.2f} ms per step")
        print(f"  {'GPU ms/step':>11} {'share':>6} {'calls':>6} {'avg us':>7}  kernel")
        for e in kern[:6]:
            print(f"  {e.self_device_time_total / steps / 1e3:>11.3f} {e.self_device_time_total / total:>6.0%} "
                  f"{e.count // steps:>6} {e.self_device_time_total / e.count:>7.1f}  {e.key[:70]}")
    else:
        print(avgs.table(sort_by="self_cpu_time_total", row_limit=8, max_name_column_width=40))

if __name__ == "__main__":
    if DEV == "cuda":
        print("GPU:", torch.cuda.get_device_name(), "| torch", torch.__version__)
    run("prefill", CTX)
    run("prefill + sync", CTX, debug_sync=True)
    run("decode", 1)
