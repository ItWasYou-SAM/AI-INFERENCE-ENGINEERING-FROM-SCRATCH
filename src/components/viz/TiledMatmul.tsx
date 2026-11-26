import { useState } from 'react';

const N = 16; // matrices are N x N cells

type Mode = 'untiled' | 'tiled';

export default function TiledMatmul() {
  const [T, setT] = useState(4); // tile size
  const [mode, setMode] = useState<Mode>('tiled');
  const [tile, setTile] = useState<[number, number]>([1, 2]); // selected C tile (row, col)
  const [step, setStep] = useState(1); // k-tiles processed so far (1 .. N / T)
  const steps = N / T;
  const s = Math.min(step, steps);
  const [tr, tc] = [Math.min(tile[0], steps - 1), Math.min(tile[1], steps - 1)];
  const k0 = (s - 1) * T; // the k-slice being worked on now

  // Global-memory loads for this block's T x T outputs after s k-steps.
  const untiledLoads = T * T * 2 * s * T; // every thread loads its own row and column elements
  const tiledLoads = 2 * T * T * s; // the block loads one A tile and one B tile per step
  const loads = mode === 'tiled' ? tiledLoads : untiledLoads;
  const flops = 2 * T * T * s * T;
  const intensity = flops / (loads * 4);

  const C = 15; // cell size
  const GAP = 46;
  const W = 3 * N * C + 2 * GAP + 20;
  const ox = [10, 10 + N * C + GAP, 10 + 2 * (N * C + GAP)]; // A, B, C origins
  const oy = 34;

  const cell = (m: 0 | 1 | 2, r: number, c: number) => {
    let fill = 'transparent';
    let op = 1;
    if (m === 0) {
      // A: row strip of the tile; current slice highlighted
      if (r >= tr * T && r < tr * T + T) {
        if (c >= k0 && c < k0 + T) fill = 'var(--amber)';
        else if (c < k0) { fill = 'var(--amber)'; op = 0.18; }
      }
    } else if (m === 1) {
      if (c >= tc * T && c < tc * T + T) {
        if (r >= k0 && r < k0 + T) fill = 'var(--blue)';
        else if (r < k0) { fill = 'var(--blue)'; op = 0.18; }
      }
    } else if (r >= tr * T && r < tr * T + T && c >= tc * T && c < tc * T + T) {
      fill = 'var(--violet)';
      op = 0.25 + 0.75 * (s / steps);
    }
    return { fill, op: fill === 'transparent' ? 1 : op * (fill === 'var(--violet)' ? 1 : 0.75) };
  };

  return (
    <div className="not-prose text-ink">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Kernel</span>
        {(['untiled', 'tiled'] as Mode[]).map((m) => (
          <button key={m} type="button" aria-pressed={m === mode} onClick={() => setMode(m)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${m === mode ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {m === 'untiled' ? 'one output per thread, no reuse' : 'shared-memory tiles'}
          </button>
        ))}
        <span className="eyebrow ml-3 mr-1">Tile</span>
        {[2, 4, 8].map((t) => (
          <button key={t} type="button" aria-pressed={t === T} onClick={() => { setT(t); setStep(1); }}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${t === T ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {t}×{t}
          </button>
        ))}
      </div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => setStep(Math.max(1, s - 1))} disabled={s === 1}
          className="rounded-md border border-rule px-2.5 py-1 font-mono text-xs disabled:opacity-40">← prev k</button>
        <button type="button" onClick={() => setStep(Math.min(steps, s + 1))} disabled={s === steps}
          className="rounded-md border border-rule px-2.5 py-1 font-mono text-xs disabled:opacity-40">next k →</button>
        <span className="font-mono text-xs text-ink-soft">k-step {s} of {steps} · click a tile of C to pick another block</span>
      </div>

      <svg viewBox={`0 0 ${W} ${oy + N * C + 30}`} className="w-full" role="img"
        aria-label={`Block computing C tile (${tr}, ${tc}) at k-step ${s} of ${steps}: ${loads} loads from memory so far.`}>
        {(['A', 'B', 'C'] as const).map((name, m) => (
          <text key={name} x={ox[m]} y={22} fontSize={12} className="fill-ink-soft font-mono">
            {name === 'A' ? 'A (rows of the tile)' : name === 'B' ? 'B (columns of the tile)' : 'C = A × B'}
          </text>
        ))}
        <text x={ox[1] - GAP / 2} y={oy + (N * C) / 2 + 6} fontSize={20} textAnchor="middle" className="fill-ink-soft">×</text>
        <text x={ox[2] - GAP / 2} y={oy + (N * C) / 2 + 6} fontSize={20} textAnchor="middle" className="fill-ink-soft">=</text>
        {[0, 1, 2].map((m) =>
          Array.from({ length: N * N }, (_, i) => {
            const r = Math.floor(i / N), c = i % N;
            const { fill, op } = cell(m as 0 | 1 | 2, r, c);
            return (
              <rect key={`${m}-${i}`} x={ox[m] + c * C + 0.5} y={oy + r * C + 0.5} width={C - 1} height={C - 1} rx={1.5}
                fill={fill} opacity={op} stroke="var(--rule)" strokeWidth={0.6}
                style={m === 2 ? { cursor: 'pointer' } : undefined}
                onClick={m === 2 ? () => { setTile([Math.floor(r / T), Math.floor(c / T)]); setStep(1); } : undefined} />
            );
          }),
        )}
        {/* tile grid lines */}
        {[0, 1, 2].map((m) =>
          Array.from({ length: steps + 1 }, (_, i) => (
            <g key={`g${m}-${i}`} stroke="var(--ink-faint)" strokeWidth={1} opacity={0.6}>
              <line x1={ox[m] + i * T * C} y1={oy} x2={ox[m] + i * T * C} y2={oy + N * C} />
              <line x1={ox[m]} y1={oy + i * T * C} x2={ox[m] + N * C} y2={oy + i * T * C} />
            </g>
          )),
        )}
        <rect x={ox[2] + tc * T * C} y={oy + tr * T * C} width={T * C} height={T * C} fill="none" stroke="var(--violet)" strokeWidth={2.2} />
        <text x={ox[0]} y={oy + N * C + 20} fontSize={11} className="fill-ink-faint font-mono">
          {mode === 'tiled'
            ? `this step: the block loads one ${T}×${T} tile of A and one of B into shared memory, once`
            : `this step: each of the ${T * T} threads loads its own ${T} values of A and ${T} of B`}
        </text>
      </svg>

      <dl className="mt-2 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">outputs in this block</dt>
          <dd className="text-lg">{T * T}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">loads from memory so far</dt>
          <dd className="text-lg" style={{ color: mode === 'tiled' ? 'var(--green)' : 'var(--red)' }}>{loads}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">{mode === 'tiled' ? 'fewer than untiled' : 'tiled would need'}</dt>
          <dd className="text-lg">{mode === 'tiled' ? `${T}×` : tiledLoads}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">FLOP per byte loaded</dt>
          <dd className="text-lg">{intensity.toFixed(2)}</dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-ink-soft">
        {mode === 'tiled'
          ? `Each value in shared memory is used by ${T} threads, so memory traffic falls by the tile width, ${T}×, and the intensity rises to T / 4 FLOP per byte. Bigger tiles mean more reuse, until shared memory and registers run out.`
          : 'Without tiling, the threads that share a row of A each load it separately. Caches catch some of the repeats, but every load is still an instruction, and the intensity stays at 0.25 FLOP per byte.'}
      </p>
    </div>
  );
}
