"""
Lab 3.5 — Derive the arithmetic intensity of a 7B model

For Llama-2-7B in BF16, count FLOPs and bytes for every operation in one forward
step, then place each on an H100 roofline. Three scenarios:

  decode  batch 1,  context 1024     (one user chatting)
  decode  batch 64, context 1024     (a busy server)
  prefill 2048-token prompt          (the first step of a request)

Everything is a formula you can check by hand. Assumes attention is fused
(FlashAttention-style: the T x T score matrix never goes to HBM).

Run:  python labs/m03/3.5/derive_ai_7b.py      (CPU only, standard library)
"""

L, D, H, KV, DH, FF, V = 32, 4096, 32, 32, 128, 11008, 32000   # Llama-2-7B
B = 2                                                           # bytes per BF16 value
PEAK, BW = 989e12, 3.35e12                                      # H100 SXM dense BF16, HBM3
RIDGE = PEAK / BW


def matmul(m, k, n):
    """[m,k] @ [k,n]: FLOPs, and bytes for reading both inputs and writing the output."""
    return 2 * m * k * n, B * (m * k + k * n + m * n)


def step_ops(batch, new_tokens, context):
    """Ops for one forward step. new_tokens per sequence; context = KV length attended to."""
    M = batch * new_tokens                                      # rows in every weight matmul
    kvd = KV * DH
    ops = {
        "QKV proj": matmul(M, D, D + 2 * kvd),
        "O proj": matmul(M, D, D),
        "MLP gate+up": matmul(M, D, 2 * FF),
        "MLP down": matmul(M, FF, D),
    }
    # attention: each query row dots `context` keys and mixes `context` values
    if new_tokens == 1:                                         # decode: read the whole cache
        flops = batch * 4 * context * D
        byts = batch * (2 * context * kvd * B + 2 * D * B)      # K and V, plus q in / out
    else:                                                       # prefill: causal, fused
        flops = batch * 4 * new_tokens * new_tokens * D // 2
        byts = batch * B * (2 * new_tokens * D + 2 * new_tokens * kvd)
    ops["attention"] = (flops, byts)
    per_layer = {k: (f * L, b * L) for k, (f, b) in ops.items()}
    per_layer["LM head"] = matmul(batch, D, V)                  # only the last position per sequence
    return per_layer


def report(title, batch, new_tokens, context):
    ops = step_ops(batch, new_tokens, context)
    print(f"\n{title}")
    print(f"{'op':<13}{'GFLOP':>10}{'GB moved':>10}{'FLOP/B':>9}{'time (us)':>11}  bound")
    tf = tb = tt = 0.0
    for name, (f, b) in ops.items():
        t = max(f / PEAK, b / BW)
        tf, tb, tt = tf + f, tb + b, tt + t
        print(f"{name:<13}{f / 1e9:>10.1f}{b / 1e9:>10.2f}{f / b:>9.1f}{t * 1e6:>11.0f}  "
              f"{'compute' if f / b > RIDGE else 'memory'}")
    print(f"{'total':<13}{tf / 1e9:>10.1f}{tb / 1e9:>10.2f}{tf / tb:>9.1f}{tt * 1e6:>11.0f}")
    toks = batch * (1 if new_tokens == 1 else new_tokens)
    print(f"-> step {tt * 1e3:.2f} ms at the roofline; {toks / tt:,.0f} tokens/s "
          f"({'per-user ' + format(1 / tt, ',.0f') + ' tok/s' if new_tokens == 1 else 'prefill'})")
    return tt


if __name__ == "__main__":
    print(f"Llama-2-7B in BF16 on an H100 SXM: ridge = {RIDGE:.0f} FLOP/byte")
    report("DECODE  batch 1, context 1024", 1, 1, 1024)
    report("DECODE  batch 64, context 1024", 64, 1, 1024)
    report("PREFILL one 2048-token prompt", 1, 2048, 2048)
    print("\nThese are lower bounds: real kernels reach maybe 60-85% of each roof.")
