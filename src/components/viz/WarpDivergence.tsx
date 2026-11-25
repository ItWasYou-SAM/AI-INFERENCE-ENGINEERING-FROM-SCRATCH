import { useState } from 'react';

const LANES = Array.from({ length: 32 }, (_, i) => i);

// Fixed pseudo-random pattern so the figure is deterministic.
const RAND = LANES.map((i) => ((i * 2654435761) >>> 0) % 7 < 3);

const PATTERNS = [
  { id: 'uniform', label: 'if (warp_id % 2)', cond: () => true, note: 'Every lane in this warp agrees, so only one path runs. Branching per warp is free.' },
  { id: 'half', label: 'if (lane < 16)', cond: (i: number) => i < 16, note: 'Half the lanes take each path. The warp runs both, one after the other.' },
  { id: 'odd', label: 'if (lane % 2)', cond: (i: number) => i % 2 === 1, note: 'Same cost as lane < 16: the hardware does not care which lanes, only that they disagree.' },
  { id: 'one', label: 'if (lane == 0)', cond: (i: number) => i === 0, note: 'One lane takes the if. The other 31 sit idle while it runs, then it sits idle during the else.' },
  { id: 'data', label: 'if (x[i] > 0)', cond: (i: number) => RAND[i], note: 'Data-dependent branches diverge unpredictably. Common in sampling, masking and sparse code.' },
] as const;

export default function WarpDivergence() {
  const [pat, setPat] = useState<(typeof PATTERNS)[number]>(PATTERNS[1]);
  const [ifLen, setIfLen] = useState(6);
  const [elseLen, setElseLen] = useState(6);

  const taken = LANES.map((i) => pat.cond(i));
  const nIf = taken.filter(Boolean).length;
  const nElse = 32 - nIf;
  const runsIf = nIf > 0;
  const runsElse = nElse > 0;
  const cycles = (runsIf ? ifLen : 0) + (runsElse ? elseLen : 0);
  const useful = nIf * ifLen + nElse * elseLen;
  const eff = useful / (32 * cycles);
  const best = Math.max(nIf ? ifLen : 0, nElse ? elseLen : 0);

  const passes = [
    runsIf && { name: 'if-body', len: ifLen, active: taken, color: 'var(--blue)' },
    runsElse && { name: 'else-body', len: elseLen, active: taken.map((t) => !t), color: 'var(--amber)' },
  ].filter(Boolean) as { name: string; len: number; active: boolean[]; color: string }[];

  const total = ifLen + elseLen; // timeline scale = worst case

  return (
    <div className="not-prose text-ink">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Branch</span>
        {PATTERNS.map((p) => (
          <button
            key={p.id}
            type="button"
            aria-pressed={p === pat}
            onClick={() => setPat(p)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${
              p === pat ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* lanes */}
      <p className="eyebrow mb-1.5">One warp: 32 lanes, one instruction stream</p>
      <div className="grid grid-cols-16 gap-1" style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }} aria-label={`${nIf} lanes take the if, ${nElse} take the else`}>
        {LANES.map((i) => (
          <div
            key={i}
            className="flex h-7 items-center justify-center rounded font-mono text-[0.6rem] transition-colors"
            style={{
              background: `color-mix(in srgb, ${taken[i] ? 'var(--blue)' : 'var(--amber)'} 30%, transparent)`,
              border: `1px solid ${taken[i] ? 'var(--blue)' : 'var(--amber)'}`,
            }}
          >
            {i}
          </div>
        ))}
      </div>
      <div className="mt-1.5 flex gap-4 font-mono text-[0.7rem]">
        <span style={{ color: 'var(--blue)' }}>■ takes if ({nIf})</span>
        <span style={{ color: 'var(--amber)' }}>■ takes else ({nElse})</span>
      </div>

      {/* timeline */}
      <p className="eyebrow mt-5 mb-2">What the warp executes, in order</p>
      <div className="space-y-2">
        {passes.map((p) => (
          <div key={p.name} className="flex items-center gap-3">
            <span className="w-20 font-mono text-xs" style={{ color: p.color }}>{p.name}</span>
            <div className="flex-1">
              <div className="h-8 rounded-sm" style={{ width: `${(p.len / total) * 100}%`, background: 'color-mix(in srgb, var(--rule) 30%, transparent)', transition: 'width 0.3s ease' }}>
                <div className="grid h-full gap-px" style={{ gridTemplateRows: 'repeat(32, 1fr)' }}>
                  {p.active.map((a, i) => (
                    <div key={i} style={{ background: a ? p.color : 'transparent', opacity: a ? 0.85 : 1 }} />
                  ))}
                </div>
              </div>
            </div>
            <span className="w-24 text-right font-mono text-xs text-ink-soft">
              {p.len} instr · {p.active.filter(Boolean).length}/32 on
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        {[
          { label: 'if-body length', v: ifLen, set: setIfLen, color: 'var(--blue)' },
          { label: 'else-body length', v: elseLen, set: setElseLen, color: 'var(--amber)' },
        ].map((s) => (
          <label key={s.label} className="block">
            <span className="eyebrow flex justify-between" style={{ color: s.color }}>
              {s.label} <b>{s.v} instructions</b>
            </span>
            <input type="range" min={1} max={20} value={s.v} onChange={(e) => s.set(+e.target.value)} className="mt-1 w-full" style={{ accentColor: s.color }} />
          </label>
        ))}
      </div>

      <div className="mt-4 rounded-md border border-rule bg-paper p-4">
        <dl className="grid grid-cols-3 gap-2 font-mono text-xs">
          <div>
            <dt className="text-ink-faint">instructions issued</dt>
            <dd className="text-base">{cycles}</dd>
          </div>
          <div>
            <dt className="text-ink-faint">if no divergence</dt>
            <dd className="text-base">{best}</dd>
          </div>
          <div>
            <dt className="text-ink-faint">lane efficiency</dt>
            <dd className="text-base" style={{ color: eff > 0.9 ? 'var(--green)' : eff > 0.6 ? 'var(--amber)' : 'var(--red)' }}>
              {(eff * 100).toFixed(0)}%
            </dd>
          </div>
        </dl>
        <p className="mt-2 text-sm text-ink-soft">{pat.note}</p>
      </div>
    </div>
  );
}
