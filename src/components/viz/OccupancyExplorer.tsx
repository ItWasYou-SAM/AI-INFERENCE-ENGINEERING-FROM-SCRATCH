import { useMemo, useState } from 'react';

/** H100 (compute capability 9.0) per-SM limits. Allocation granularity is ignored for clarity. */
const SM = { maxWarps: 64, maxBlocks: 32, regs: 65536, smemKB: 228, schedulers: 4 };

function occupancy(tpb: number, regs: number, smemKB: number) {
  const wpb = Math.ceil(tpb / 32);
  const limits = [
    { name: 'warp slots', blocks: Math.floor(SM.maxWarps / wpb) },
    { name: 'block slots', blocks: SM.maxBlocks },
    { name: 'registers', blocks: Math.floor(SM.regs / (regs * tpb)) },
    { name: 'shared memory', blocks: smemKB > 0 ? Math.floor(SM.smemKB / smemKB) : SM.maxBlocks },
  ];
  const blocks = Math.min(...limits.map((l) => l.blocks));
  return { limits, blocks, warps: blocks * wpb, wpb };
}

/** Cycle-level sketch of one scheduler: each warp issues `compute` instructions, then waits `latency` cycles on a load. */
function simulate(nWarps: number, compute: number, latency: number, horizon: number) {
  const readyAt = Array.from({ length: nWarps }, (_, w) => w); // stagger starts slightly
  const left = Array(nWarps).fill(compute);
  const issued: number[][] = Array.from({ length: nWarps }, () => []);
  let busy = 0;
  let rr = 0;
  for (let t = 0; t < horizon; t++) {
    for (let k = 0; k < nWarps; k++) {
      const w = (rr + k) % nWarps;
      if (readyAt[w] <= t) {
        issued[w].push(t);
        busy++;
        left[w]--;
        if (left[w] === 0) {
          left[w] = compute;
          readyAt[w] = t + 1 + latency;
        }
        rr = (w + 1) % nWarps;
        break;
      }
    }
  }
  return { issued, util: busy / horizon };
}

/** Merge consecutive cycles into [start, len] runs for drawing. */
const runs = (cycles: number[]) => {
  const out: [number, number][] = [];
  for (const c of cycles) {
    const last = out[out.length - 1];
    if (last && last[0] + last[1] === c) last[1]++;
    else out.push([c, 1]);
  }
  return out;
};

const W = 640;
const ROW = 12;

export default function OccupancyExplorer() {
  const [tpb, setTpb] = useState(256);
  const [regs, setRegs] = useState(64);
  const [smem, setSmem] = useState(0);
  const [compute, setCompute] = useState(20);
  const [latency, setLatency] = useState(500);

  const occ = occupancy(tpb, regs, smem);
  const perSched = Math.max(occ.warps > 0 ? 1 : 0, Math.floor(occ.warps / SM.schedulers));
  const horizon = 2 * (compute + latency);
  const sim = useMemo(() => simulate(perSched, compute, latency, horizon), [perSched, compute, latency, horizon]);
  const limiter = occ.limits.reduce((a, b) => (b.blocks < a.blocks ? b : a));
  const H = Math.max(1, perSched) * ROW + 24;
  const x = (t: number) => (t / horizon) * W;

  const sliders = [
    { label: 'threads per block', v: tpb, set: setTpb, min: 32, max: 1024, step: 32, fmt: (v: number) => `${v}` },
    { label: 'registers per thread', v: regs, set: setRegs, min: 16, max: 255, step: 1, fmt: (v: number) => `${v}` },
    { label: 'shared memory per block', v: smem, set: setSmem, min: 0, max: 228, step: 4, fmt: (v: number) => `${v} KB` },
  ];

  return (
    <div className="not-prose text-ink">
      <div className="grid gap-5 md:grid-cols-2">
        <div className="space-y-3">
          <p className="eyebrow">1 · What one block of your kernel needs</p>
          {sliders.map((s) => (
            <label key={s.label} className="block">
              <span className="eyebrow flex justify-between">
                {s.label} <b className="text-ink">{s.fmt(s.v)}</b>
              </span>
              <input type="range" min={s.min} max={s.max} step={s.step} value={s.v} onChange={(e) => s.set(+e.target.value)} className="mt-1 w-full" />
            </label>
          ))}
        </div>
        <div className="rounded-md border border-rule bg-paper p-4">
          <p className="eyebrow">2 · How many blocks fit on one SM</p>
          <table className="mt-2 w-full font-mono text-xs">
            <tbody>
              {occ.limits.map((l) => (
                <tr key={l.name} style={{ color: l === limiter ? 'var(--red)' : undefined }}>
                  <td className="py-0.5">{l.name}</td>
                  <td className="py-0.5 text-right">{l.blocks} blocks</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 font-mono text-xs">
            {occ.blocks} blocks × {occ.wpb} warps = <b className="text-base">{occ.warps}</b> / 64 warps
          </p>
          <div className="mt-2 h-3 overflow-hidden rounded-sm bg-[color-mix(in_srgb,var(--rule)_40%,transparent)]">
            <div className="h-full" style={{ width: `${(occ.warps / SM.maxWarps) * 100}%`, background: 'var(--blue)', transition: 'width 0.3s ease' }} />
          </div>
          <p className="mt-2 text-sm text-ink-soft">
            Occupancy <b className="text-ink">{((occ.warps / SM.maxWarps) * 100).toFixed(0)}%</b>
            {occ.warps === 0 ? ': the block does not fit at all.' : <>, limited by <b style={{ color: 'var(--red)' }}>{limiter.name}</b>.</>}
          </p>
        </div>
      </div>

      <div className="mt-6">
        <p className="eyebrow mb-2">3 · One of the 4 schedulers, with its {perSched} warp{perSched === 1 ? '' : 's'}</p>
        <div className="mb-3 grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="eyebrow flex justify-between" style={{ color: 'var(--blue)' }}>
              math between loads <b>{compute} cycles</b>
            </span>
            <input type="range" min={2} max={100} value={compute} onChange={(e) => setCompute(+e.target.value)} className="mt-1 w-full" style={{ accentColor: 'var(--blue)' }} />
          </label>
          <label className="block">
            <span className="eyebrow flex justify-between" style={{ color: 'var(--red)' }}>
              memory latency <b>{latency} cycles</b>
            </span>
            <input type="range" min={50} max={800} step={10} value={latency} onChange={(e) => setLatency(+e.target.value)} className="mt-1 w-full" style={{ accentColor: 'var(--red)' }} />
          </label>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Scheduler busy ${(sim.util * 100).toFixed(0)} percent of cycles`}>
          {sim.issued.map((cyc, w) => (
            <g key={w} transform={`translate(0 ${w * ROW})`}>
              <rect x={0} y={1} width={W} height={ROW - 3} fill="var(--red)" opacity={0.08} />
              {runs(cyc).map(([s, l]) => (
                <rect key={s} x={x(s)} y={1} width={Math.max(1, x(l))} height={ROW - 3} fill="var(--blue)" />
              ))}
            </g>
          ))}
          <text x={0} y={H - 6} fontSize={10} className="fill-ink-faint font-mono">
            0
          </text>
          <text x={W} y={H - 6} fontSize={10} textAnchor="end" className="fill-ink-faint font-mono">
            {horizon} cycles
          </text>
        </svg>
        <p className="mt-2 text-sm text-ink-soft">
          Blue = the warp is issuing instructions; pale red = waiting on memory. The scheduler is busy{' '}
          <b style={{ color: sim.util > 0.9 ? 'var(--green)' : sim.util > 0.5 ? 'var(--amber)' : 'var(--red)' }}>{(sim.util * 100).toFixed(0)}%</b> of
          cycles. Full speed needs about <b className="text-ink">{Math.ceil((compute + latency) / compute)}</b> warps per scheduler.
        </p>
      </div>
    </div>
  );
}
