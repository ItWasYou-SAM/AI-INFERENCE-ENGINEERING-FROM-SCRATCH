import { useState } from 'react';

const N = 32; // elements in shared memory, one per thread
const WARP = 8; // drawn as 8-lane warps so four fit across; real warps have 32 lanes
const INIT = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5];

type Op = { tid: number; dst: number; src: number };

const STRATEGIES = [
  {
    id: 'v1',
    label: 'v1 interleaved, divergent',
    code: 'if (tid % (2*s) == 0) s[tid] += s[tid + s];',
    strides: [1, 2, 4, 8, 16],
    ops: (s: number): Op[] =>
      Array.from({ length: N }, (_, t) => t)
        .filter((t) => t % (2 * s) === 0)
        .map((t) => ({ tid: t, dst: t, src: t + s })),
    note: 'Working threads are scattered: every warp keeps a few busy lanes, so every warp has to issue the instruction while most of its lanes sit idle.',
  },
  {
    id: 'v2',
    label: 'v2 interleaved, compact',
    code: 'idx = 2*s*tid; if (idx < N) s[idx] += s[idx + s];',
    strides: [1, 2, 4, 8, 16],
    ops: (s: number): Op[] =>
      Array.from({ length: N }, (_, t) => t)
        .filter((t) => 2 * s * t < N)
        .map((t) => ({ tid: t, dst: 2 * s * t, src: 2 * s * t + s })),
    note: 'Same additions, but done by threads 0, 1, 2, … so whole warps go idle. The cost moves into memory: neighbouring lanes now touch addresses 2s apart, which causes shared-memory bank conflicts (6.4).',
  },
  {
    id: 'v3',
    label: 'v3 sequential',
    code: 'if (tid < s) s[tid] += s[tid + s];',
    strides: [16, 8, 4, 2, 1],
    ops: (s: number): Op[] => Array.from({ length: s }, (_, t) => ({ tid: t, dst: t, src: t + s })),
    note: 'Fold the top half onto the bottom half. Working threads are contiguous and so are their addresses: whole warps retire early and no lanes collide on a bank.',
  },
] as const;

function states(strat: (typeof STRATEGIES)[number]) {
  const out = [INIT.slice()];
  for (const s of strat.strides) {
    const next = out[out.length - 1].slice();
    for (const op of strat.ops(s)) next[op.dst] = next[op.dst] + next[op.src];
    out.push(next);
  }
  return out;
}

export default function ReductionTree() {
  const [strat, setStrat] = useState<(typeof STRATEGIES)[number]>(STRATEGIES[0]);
  const [step, setStep] = useState(0);
  const all = states(strat);
  const s = strat.strides[step];
  const ops = strat.ops(s);
  const before = all[step];
  const after = all[step + 1];
  const active = new Set(ops.map((o) => o.tid));
  const dsts = new Set(ops.map((o) => o.dst));
  const srcs = new Set(ops.map((o) => o.src));
  const warps = Array.from({ length: N / WARP }, (_, w) => Array.from({ length: WARP }, (_, l) => active.has(w * WARP + l)));
  const busyWarps = warps.filter((w) => w.some(Boolean)).length;
  const laneUse = ops.length / (busyWarps * WARP);
  // over the whole reduction
  const totalWarpSteps = strat.strides.reduce((acc, st) => {
    const a = new Set(strat.ops(st).map((o) => o.tid));
    return acc + Array.from({ length: N / WARP }, (_, w) => w).filter((w) => Array.from({ length: WARP }, (_, l) => a.has(w * WARP + l)).some(Boolean)).length;
  }, 0);

  const W = 640;
  const cw = (W - 40) / N;
  const cx = (i: number) => 20 + i * cw;
  const Y0 = 22, Y1 = 112, CH = 22;

  const pick = (v: (typeof STRATEGIES)[number]) => { setStrat(v); setStep(0); };

  return (
    <div className="not-prose text-ink">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Kernel</span>
        {STRATEGIES.map((v) => (
          <button key={v.id} type="button" aria-pressed={v === strat} onClick={() => pick(v)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${v === strat ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {v.label}
          </button>
        ))}
      </div>
      <p className="mb-3 font-mono text-xs text-ink-soft">{strat.code}</p>

      <div className="mb-2 flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
          className="rounded-md border border-rule px-2.5 py-1 font-mono text-xs disabled:opacity-40">← prev</button>
        <button type="button" onClick={() => setStep(Math.min(strat.strides.length - 1, step + 1))} disabled={step === strat.strides.length - 1}
          className="rounded-md border border-rule px-2.5 py-1 font-mono text-xs disabled:opacity-40">next →</button>
        <span className="font-mono text-xs text-ink-soft">step {step + 1} of {strat.strides.length} · s = {s}</span>
      </div>

      <svg viewBox={`0 0 ${W} 230`} className="w-full" role="img"
        aria-label={`Step ${step + 1}: ${ops.length} threads add pairs at distance ${s}; ${busyWarps} of ${N / WARP} warps issue the instruction.`}>
        <text x={20} y={14} fontSize={10.5} className="fill-ink-soft font-mono">shared memory before this step</text>
        {before.map((v, i) => (
          <g key={`b${i}`}>
            <rect x={cx(i) + 1} y={Y0} width={cw - 2} height={CH} rx={2}
              fill={srcs.has(i) ? 'color-mix(in srgb, var(--amber) 35%, transparent)' : dsts.has(i) ? 'color-mix(in srgb, var(--blue) 25%, transparent)' : 'transparent'}
              stroke="var(--rule)" />
            <text x={cx(i) + cw / 2} y={Y0 + 15} fontSize={9} textAnchor="middle" className="fill-ink font-mono">{v}</text>
          </g>
        ))}
        {ops.map((o) => (
          <g key={`op${o.tid}`}>
            <line x1={cx(o.src) + cw / 2} y1={Y0 + CH} x2={cx(o.dst) + cw / 2} y2={Y1} stroke="var(--amber)" strokeWidth={1.2} opacity={0.8} />
            <line x1={cx(o.dst) + cw / 2} y1={Y0 + CH} x2={cx(o.dst) + cw / 2} y2={Y1} stroke="var(--blue)" strokeWidth={1.2} opacity={0.8} />
          </g>
        ))}
        {after.map((v, i) => (
          <g key={`a${i}`}>
            <rect x={cx(i) + 1} y={Y1} width={cw - 2} height={CH} rx={2}
              fill={dsts.has(i) ? 'color-mix(in srgb, var(--blue) 35%, transparent)' : 'transparent'}
              stroke="var(--rule)" opacity={dsts.has(i) ? 1 : 0.6} />
            <text x={cx(i) + cw / 2} y={Y1 + 15} fontSize={9} textAnchor="middle" className="fill-ink font-mono" opacity={dsts.has(i) ? 1 : 0.5}>{v}</text>
          </g>
        ))}
        <text x={20} y={Y1 + CH + 14} fontSize={10.5} className="fill-ink-soft font-mono">after</text>

        <text x={20} y={172} fontSize={10.5} className="fill-ink-soft font-mono">threads 0–31, drawn as 4 warps of {WARP}</text>
        {warps.map((lanes, w) => {
          const busy = lanes.some(Boolean);
          return (
            <g key={`w${w}`}>
              <rect x={cx(w * WARP) + 0.5} y={180} width={WARP * cw - 1} height={30} rx={4}
                fill="transparent" stroke={busy ? 'var(--ink)' : 'var(--rule)'} strokeDasharray={busy ? undefined : '3 3'} />
              {lanes.map((on, l) => (
                <circle key={l} cx={cx(w * WARP + l) + cw / 2} cy={195} r={5.5}
                  fill={on ? 'var(--blue)' : 'transparent'} stroke={on ? 'var(--blue)' : 'var(--ink-faint)'} strokeWidth={1} />
              ))}
              <text x={cx(w * WARP) + (WARP * cw) / 2} y={224} fontSize={9.5} textAnchor="middle"
                className={busy ? 'fill-ink font-mono' : 'fill-ink-faint font-mono'}>
                warp {w}: {busy ? 'issues' : 'idle'}
              </text>
            </g>
          );
        })}
      </svg>

      <dl className="mt-2 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">threads adding</dt>
          <dd className="text-lg">{ops.length} / {N}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">warps issuing</dt>
          <dd className="text-lg">{busyWarps} / {N / WARP}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">lanes used in busy warps</dt>
          <dd className="text-lg" style={{ color: laneUse > 0.9 ? 'var(--green)' : laneUse > 0.4 ? 'var(--amber)' : 'var(--red)' }}>{(laneUse * 100).toFixed(0)}%</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">warp-steps, whole sum</dt>
          <dd className="text-lg">{totalWarpSteps}</dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-ink-soft">{strat.note}</p>
    </div>
  );
}
