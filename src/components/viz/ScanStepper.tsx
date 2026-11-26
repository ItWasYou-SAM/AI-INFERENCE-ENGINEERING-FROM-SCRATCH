import { useState } from 'react';

const N = 16;
const INIT = [3, 1, 7, 0, 4, 1, 6, 3, 2, 5, 0, 4, 1, 2, 6, 1];

type Op = { dst: number; src?: number; kind: 'add' | 'swap' | 'zero' };
type Step = { label: string; phase: string; ops: Op[]; apply: (a: number[]) => number[] };

function hillisSteele(): Step[] {
  const steps: Step[] = [];
  for (let off = 1; off < N; off *= 2) {
    const ops: Op[] = [];
    for (let i = off; i < N; i++) ops.push({ dst: i, src: i - off, kind: 'add' });
    steps.push({
      label: `offset ${off}`,
      phase: 'scan',
      ops,
      apply: (a) => a.map((v, i) => (i >= off ? v + a[i - off] : v)),
    });
  }
  return steps;
}

function blelloch(): Step[] {
  const steps: Step[] = [];
  let off = 1;
  for (let d = N / 2; d >= 1; d /= 2, off *= 2) {
    const o = off;
    const ops: Op[] = Array.from({ length: d }, (_, t) => ({ src: o * (2 * t + 1) - 1, dst: o * (2 * t + 2) - 1, kind: 'add' as const }));
    steps.push({
      label: `up ${d}`,
      phase: 'up-sweep',
      ops,
      apply: (a) => { const b = a.slice(); for (const op of ops) b[op.dst] += a[op.src!]; return b; },
    });
  }
  steps.push({ label: 'clear root', phase: 'clear', ops: [{ dst: N - 1, kind: 'zero' }], apply: (a) => { const b = a.slice(); b[N - 1] = 0; return b; } });
  for (let d = 1; d < N; d *= 2) {
    off /= 2;
    const o = off;
    const ops: Op[] = Array.from({ length: d }, (_, t) => ({ src: o * (2 * t + 1) - 1, dst: o * (2 * t + 2) - 1, kind: 'swap' as const }));
    steps.push({
      label: `down ${d}`,
      phase: 'down-sweep',
      ops,
      apply: (a) => {
        const b = a.slice();
        for (const op of ops) { const left = a[op.src!]; b[op.src!] = a[op.dst]; b[op.dst] = a[op.dst] + left; }
        return b;
      },
    });
  }
  return steps;
}

const ALGOS = [
  {
    id: 'hs',
    label: 'Hillis–Steele',
    steps: hillisSteele(),
    result: 'inclusive',
    note: 'Every element adds the one "offset" to its left; the offset doubles each step. Few steps (log₂ n), but n − offset additions in each: about n·log₂ n in total. Good when there are more threads than work, as inside a warp.',
  },
  {
    id: 'bl',
    label: 'Blelloch',
    steps: blelloch(),
    result: 'exclusive',
    note: 'An up-sweep (a reduction tree) builds partial sums; a down-sweep pushes prefixes back down. Twice as many steps, but only about 2n additions in total: work-efficient. Good when work, not latency, is the limit.',
  },
] as const;

export default function ScanStepper() {
  const [algo, setAlgo] = useState<(typeof ALGOS)[number]>(ALGOS[0]);
  const [step, setStep] = useState(0); // 0 = input, k = after k steps
  const states = [INIT.slice()];
  for (const s of algo.steps) states.push(s.apply(states[states.length - 1]));
  const cur = step > 0 ? algo.steps[step - 1] : null;
  const before = states[Math.max(0, step - 1)];
  const after = states[step];
  const addsSoFar = algo.steps.slice(0, step).reduce((n, s) => n + s.ops.filter((o) => o.kind !== 'zero').length, 0);
  const addsTotal = algo.steps.reduce((n, s) => n + s.ops.filter((o) => o.kind !== 'zero').length, 0);
  const done = step === algo.steps.length;
  const inclusive = INIT.reduce<number[]>((acc, v) => [...acc, (acc[acc.length - 1] ?? 0) + v], []);
  const expected = algo.result === 'inclusive' ? inclusive : inclusive.map((v, i) => v - INIT[i]);

  const dsts = new Set(cur?.ops.map((o) => o.dst) ?? []);
  const srcs = new Set(cur?.ops.flatMap((o) => (o.src !== undefined ? [o.src] : [])) ?? []);

  const W = 640;
  const cw = (W - 40) / N;
  const cx = (i: number) => 20 + i * cw;
  const Y0 = 26, Y1 = 120, CH = 26;

  const pick = (a: (typeof ALGOS)[number]) => { setAlgo(a); setStep(0); };

  return (
    <div className="not-prose text-ink">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Algorithm</span>
        {ALGOS.map((a) => (
          <button key={a.id} type="button" aria-pressed={a === algo} onClick={() => pick(a)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${a === algo ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {a.label}
          </button>
        ))}
      </div>

      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <button type="button" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
          className="mr-1 rounded-md border border-rule px-2.5 py-1 font-mono text-xs disabled:opacity-40">← prev</button>
        <button type="button" onClick={() => setStep(Math.min(algo.steps.length, step + 1))} disabled={done}
          className="mr-2 rounded-md border border-rule px-2.5 py-1 font-mono text-xs disabled:opacity-40">next →</button>
        {algo.steps.map((s, i) => (
          <button key={i} type="button" onClick={() => setStep(i + 1)}
            className={`rounded px-1.5 py-0.5 font-mono text-[0.65rem] transition-colors ${i + 1 === step ? 'bg-ink text-paper' : i + 1 < step ? 'text-ink' : 'text-ink-faint'}`}
            style={{ border: '1px solid var(--rule)' }}>
            {s.label}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} 170`} className="w-full" role="img"
        aria-label={cur ? `${cur.phase}, ${cur.label}: ${cur.ops.length} operations` : 'The input array'}>
        <text x={20} y={16} fontSize={10.5} className="fill-ink-soft font-mono">
          {cur ? `before: ${cur.phase}, ${cur.label}` : 'input'}
        </text>
        {before.map((v, i) => (
          <g key={`b${i}`}>
            <rect x={cx(i) + 1.5} y={Y0} width={cw - 3} height={CH} rx={3}
              fill={srcs.has(i) ? 'color-mix(in srgb, var(--amber) 35%, transparent)' : dsts.has(i) ? 'color-mix(in srgb, var(--blue) 22%, transparent)' : 'transparent'}
              stroke="var(--rule)" />
            <text x={cx(i) + cw / 2} y={Y0 + 17} fontSize={11} textAnchor="middle" className="fill-ink font-mono">{v}</text>
          </g>
        ))}
        {cur?.ops.map((o, k) => (
          <g key={k}>
            {o.src !== undefined && (
              <line x1={cx(o.src) + cw / 2} y1={Y0 + CH} x2={cx(o.dst) + cw / 2} y2={Y1} stroke="var(--amber)" strokeWidth={1.4} />
            )}
            {o.kind === 'swap' && (
              <line x1={cx(o.dst) + cw / 2} y1={Y0 + CH} x2={cx(o.src!) + cw / 2} y2={Y1} stroke="var(--violet)" strokeWidth={1.4} strokeDasharray="4 3" />
            )}
            <line x1={cx(o.dst) + cw / 2} y1={Y0 + CH} x2={cx(o.dst) + cw / 2} y2={Y1} stroke={o.kind === 'zero' ? 'var(--red)' : 'var(--blue)'} strokeWidth={1.4} />
          </g>
        ))}
        {step === 0 && (
          <text x={W / 2} y={Y1 + 17} fontSize={11} textAnchor="middle" className="fill-ink-faint font-mono">
            press next → to run the first parallel step
          </text>
        )}
        {step > 0 && after.map((v, i) => {
          const changed = dsts.has(i) || (cur?.ops.some((o) => o.kind === 'swap' && o.src === i) ?? false);
          return (
            <g key={`a${i}`}>
              <rect x={cx(i) + 1.5} y={Y1} width={cw - 3} height={CH} rx={3}
                fill={changed ? 'color-mix(in srgb, var(--blue) 35%, transparent)' : 'transparent'}
                stroke={done && v === expected[i] ? 'var(--green)' : 'var(--rule)'} />
              <text x={cx(i) + cw / 2} y={Y1 + 17} fontSize={11} textAnchor="middle" className="fill-ink font-mono">{v}</text>
            </g>
          );
        })}
        <text x={20} y={Y1 + CH + 16} fontSize={10.5} className="fill-ink-soft font-mono">
          {done ? `done: the ${algo.result} scan` : step > 0 ? 'after' : ''}
        </text>
        <text x={W - 20} y={Y1 + CH + 16} fontSize={10.5} textAnchor="end" className="font-mono">
          <tspan fill="var(--amber)">— read </tspan>
          <tspan fill="var(--blue)">— write </tspan>
          {algo.id === 'bl' && <tspan fill="var(--violet)">-- move left</tspan>}
        </text>
      </svg>

      <dl className="mt-2 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">parallel steps</dt>
          <dd className="text-lg">{step} / {algo.steps.length}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">additions so far</dt>
          <dd className="text-lg">{addsSoFar}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">additions, total</dt>
          <dd className="text-lg" style={{ color: addsTotal > 2 * N ? 'var(--amber)' : 'var(--green)' }}>{addsTotal}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">serial loop</dt>
          <dd className="text-lg">{N - 1} steps, {N - 1} adds</dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-ink-soft">{algo.note}</p>
    </div>
  );
}
