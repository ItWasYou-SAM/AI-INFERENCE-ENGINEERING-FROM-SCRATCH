"""Print one step of an Nsight Systems timeline as text: kernels, copies and NVTX ranges.

    nsys export -t sqlite -o tools.sqlite tools.nsys-rep
    python3 step_timeline.py tools.sqlite [step number, default 6] [range name, default step]

Times are in ms from the start of the chosen NVTX range. It is the same information as
the timeline view in nsys-ui, for when you have no display or want to paste it somewhere.
"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
which = int(sys.argv[2]) if len(sys.argv) > 2 else 6
outer = sys.argv[3] if len(sys.argv) > 3 else "step"

names = dict(db.execute("select id, value from StringIds"))
steps = list(db.execute("select start, end from NVTX_EVENTS where text = ? order by start", (outer,)))
a, b = steps[which]
kinds = {1: "host to device", 2: "device to host", 8: "device to device"}
ev = [(s, e, "GPU kernel  " + names[n])
      for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ?", (a, b))]
ev += [(s, e, "GPU copy    " + kinds.get(k, f"kind {k}"))
       for s, e, k in db.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ? and start < ?", (a, b))]
ev += [(s, e, "CPU range   " + t)
       for s, e, t in db.execute("select start, end, text from NVTX_EVENTS where start >= ? and start < ? and text != ? and end is not null",
                                 (a, b, outer))]
print(f"{outer} #{which}: {(b - a) / 1e6:.2f} ms")
for s, e, n in sorted(ev):
    print(f"  {(s - a) / 1e6:7.2f} - {(e - a) / 1e6:6.2f} ms  {n}")
