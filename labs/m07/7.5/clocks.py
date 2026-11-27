"""Median SM clock during each version's kernels, from an Nsight Systems GPU-metrics capture.

    nsys profile -t cuda,nvtx --gpu-metrics-devices=0 -o gm ./mlp
    nsys export -t sqlite -o gm.sqlite gm.nsys-rep
    python3 clocks.py gm.sqlite

Each kernel is assigned to the NVTX range (v0 ... v4) that was open on the CPU when it
was launched; the GPU's sampled "GPC Clock Frequency" is averaged over the kernel's run.
"""
import bisect
import collections
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
clock_id = next(i for i, n in db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS") if n.startswith("GPC Clock"))
clk = sorted(db.execute("select timestamp, value from GPU_METRICS where metricId = ?", (clock_id,)))
ts = [t for t, _ in clk]
ranges = list(db.execute("select start, end, text from NVTX_EVENTS where text like 'v%' and end is not null"))
launch = dict(db.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))


def version(corr):
    t = launch.get(corr)
    return next((name for a, b, name in ranges if t is not None and a <= t <= b), None)


per = collections.defaultdict(list)
for s, e, corr in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"):
    i, j = bisect.bisect_left(ts, s), bisect.bisect_right(ts, e)
    v = version(corr)
    if v and j > i:
        per[v].append(statistics.mean(x for _, x in clk[i:j]))

for v in sorted(per):
    mhz = statistics.median(per[v]) / 1e6   # the export stores Hz
    print(f"  {v:26} {mhz:6.0f} MHz   ({len(per[v])} kernels)")
