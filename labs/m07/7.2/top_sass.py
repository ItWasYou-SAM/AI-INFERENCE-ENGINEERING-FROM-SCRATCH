"""Print the SASS instructions where a kernel's warps spent the most stall samples.

Reads the CSV that Nsight Compute writes for its source page:

    ncu -i report.ncu-repz --page source --print-source sass --csv | python3 top_sass.py

For each of the top instructions it shows the share of all stall samples, and for
global loads and stores the L2 sectors moved against the ideal number for that access.
"""
import csv
import sys

rows = list(csv.reader(sys.stdin))
header_at = next(i for i, r in enumerate(rows) if r and r[0] == "Address")
head, body = rows[header_at], [r for r in rows[header_at + 1:] if len(r) == len(rows[header_at])]
col = {name: i for i, name in enumerate(head)}


def num(r, name):
    v = r[col[name]].replace(",", "") if name in col else ""
    try:
        return float(v)
    except ValueError:
        return 0.0


total = sum(num(r, "Warp Stall Sampling (All Samples)") for r in body) or 1.0
top = sorted(body, key=lambda r: -num(r, "Warp Stall Sampling (All Samples)"))[: int(sys.argv[1]) if len(sys.argv) > 1 else 6]
print(f"{'stall samples':>14}  {'L2 sectors / ideal':>19}  instruction")
for r in top:
    share = num(r, "Warp Stall Sampling (All Samples)") / total
    sec, ideal = num(r, "L2 Theoretical Sectors Global"), num(r, "L2 Theoretical Sectors Global Ideal")
    ratio = f"{sec / ideal:>8.1f}x" if ideal else ""
    print(f"{share:>13.1%}  {ratio:>19}  {r[col['Source']].strip()}")
