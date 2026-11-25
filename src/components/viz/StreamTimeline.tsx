import { useState } from 'react';

type Seg = { lane: number; start: number; len: number; label: string; color: string };

const LANES = ['CPU', 'copy H→D', 'compute', 'copy D→H'];
const C_H2D = 'var(--blue)';
const C_K = 'var(--violet)';
const C_D2H = 'var(--teal)';
const LAUNCH = 0.4; // CPU time to enqueue one call (ms, illustrative)

/** Schedule chunked copy → kernel → copy across `chunks` streams, with one copy engine per direction. */
function schedule(mode: 'sync' | 'streams', chunks: number, h2d: number, k: number, d2h: number) {
  const segs: Seg[] = [];
  if (mode === 'sync') {
    // cudaMemcpy blocks the CPU; the launch is async but the next cudaMemcpy waits for it.
    let t = 0;
    segs.push({ lane: 0, start: t, len: h2d, label: 'cudaMemcpy (blocked)', color: 'var(--ink-faint)' });
    segs.push({ lane: 1, start: t, len: h2d, label: 'H→D', color: C_H2D });
    t += h2d;
    segs.push({ lane: 0, start: t, len: LAUNCH, label: 'launch', color: 'var(--ink)' });
    segs.push({ lane: 2, start: t + LAUNCH, len: k, label: 'kernel', color: C_K });
    segs.push({ lane: 0, start: t + LAUNCH, len: k, label: 'cudaMemcpy waits', color: 'var(--ink-faint)' });
    t += LAUNCH + k;
    segs.push({ lane: 0, start: t, len: d2h, label: 'cudaMemcpy (blocked)', color: 'var(--ink-faint)' });
    segs.push({ lane: 3, start: t, len: d2h, label: 'D→H', color: C_D2H });
    return { segs, end: t + d2h };
  }
  let cpu = 0;
  let copyIn = 0;
  let comp = 0;
  let copyOut = 0;
  let end = 0;
  for (let i = 0; i < chunks; i++) {
    const hi = h2d / chunks;
    const ki = k / chunks;
    const di = d2h / chunks;
    // CPU enqueues three async calls per chunk and moves on.
    for (let j = 0; j < 3; j++) {
      segs.push({ lane: 0, start: cpu, len: LAUNCH * 0.8, label: '', color: 'var(--ink)' });
      cpu += LAUNCH;
    }
    const enq = cpu - 3 * LAUNCH;
    const hs = Math.max(copyIn, enq);
    segs.push({ lane: 1, start: hs, len: hi, label: `${i}`, color: C_H2D });
    copyIn = hs + hi;
    const ks = Math.max(comp, copyIn, enq + LAUNCH);
    segs.push({ lane: 2, start: ks, len: ki, label: `${i}`, color: C_K });
    comp = ks + ki;
    const ds = Math.max(copyOut, comp);
    segs.push({ lane: 3, start: ds, len: di, label: `${i}`, color: C_D2H });
    copyOut = ds + di;
    end = Math.max(end, copyOut);
  }
  segs.push({ lane: 0, start: cpu, len: Math.max(0, end - cpu), label: 'free for other work → then sync', color: 'var(--green)' });
  return { segs, end };
}

const W = 640;
const LANE_H = 30;
const LEFT = 78;

export default function StreamTimeline() {
  const [mode, setMode] = useState<'sync' | 'streams'>('streams');
  const [chunks, setChunks] = useState(4);
  const [h2d, setH2d] = useState(8);
  const [k, setK] = useState(6);
  const [d2h, setD2h] = useState(8);

  const sync = schedule('sync', 1, h2d, k, d2h);
  const cur = mode === 'sync' ? sync : schedule('streams', chunks, h2d, k, d2h);
  const scale = (W - LEFT - 10) / Math.max(sync.end, cur.end);
  const H = LANES.length * LANE_H + 28;

  return (
    <div className="not-prose text-ink">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {(['sync', 'streams'] as const).map((m) => (
          <button key={m} type="button" aria-pressed={m === mode} onClick={() => setMode(m)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${m === mode ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {m === 'sync' ? 'cudaMemcpy + launch (one stream)' : 'async, chunked over streams'}
          </button>
        ))}
        {mode === 'streams' && (
          <label className="ml-auto flex items-center gap-2 font-mono text-xs">
            streams <input type="range" min={1} max={8} value={chunks} onChange={(e) => setChunks(+e.target.value)} /> <b>{chunks}</b>
          </label>
        )}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Total ${cur.end.toFixed(1)} ms versus ${sync.end.toFixed(1)} ms serial`}>
        {LANES.map((l, i) => (
          <g key={l}>
            <text x={0} y={i * LANE_H + 19} fontSize={11} className="fill-ink-soft font-mono">{l}</text>
            <line x1={LEFT} x2={W} y1={i * LANE_H + LANE_H - 2} y2={i * LANE_H + LANE_H - 2} stroke="var(--rule)" strokeDasharray="2 4" />
          </g>
        ))}
        {cur.segs.map((s, i) => (
          <g key={i}>
            <rect x={LEFT + s.start * scale} y={s.lane * LANE_H + 5} width={Math.max(1.5, s.len * scale - 1)} height={LANE_H - 12} rx={3}
              fill={s.color} opacity={s.lane === 0 && s.color !== 'var(--ink)' ? 0.25 : 0.85} />
            {s.label && s.len * scale > 14 && (
              <text x={LEFT + s.start * scale + (s.len * scale) / 2} y={s.lane * LANE_H + 21} textAnchor="middle" fontSize={10}
                className="font-mono" fill={s.lane === 0 ? 'var(--ink)' : 'var(--paper)'}>
                {s.len * scale > s.label.length * 6.5 ? s.label : ''}
              </text>
            )}
          </g>
        ))}
        <line x1={LEFT + sync.end * scale} x2={LEFT + sync.end * scale} y1={0} y2={H - 22} stroke="var(--red)" strokeDasharray="4 3" />
        <text x={LEFT + sync.end * scale - 4} y={H - 8} textAnchor="end" fontSize={10} className="fill-red font-mono">serial {sync.end.toFixed(1)} ms</text>
        {mode === 'streams' && (
          <text x={LEFT + cur.end * scale} y={H - 8} textAnchor="middle" fontSize={10} className="fill-green font-mono">{cur.end.toFixed(1)} ms</text>
        )}
      </svg>

      <div className="mt-3 grid gap-4 sm:grid-cols-3">
        {[
          { label: 'copy in', v: h2d, set: setH2d, c: C_H2D },
          { label: 'kernel', v: k, set: setK, c: C_K },
          { label: 'copy out', v: d2h, set: setD2h, c: C_D2H },
        ].map((s) => (
          <label key={s.label} className="block">
            <span className="eyebrow flex justify-between" style={{ color: s.c }}>{s.label} <b>{s.v} ms</b></span>
            <input type="range" min={1} max={20} value={s.v} onChange={(e) => s.set(+e.target.value)} className="mt-1 w-full" style={{ accentColor: s.c }} />
          </label>
        ))}
      </div>
      <p className="mt-3 text-sm text-ink-soft">
        {mode === 'sync' ? (
          <>Each step waits for the one before, and the CPU is stuck inside <code>cudaMemcpy</code> the whole time. Only one engine is busy at any moment.</>
        ) : (
          <>
            With {chunks} stream{chunks > 1 ? 's' : ''}, chunk <i>k</i>'s kernel runs while chunk <i>k+1</i> copies in and chunk <i>k−1</i> copies out:{' '}
            <b className="text-ink">{(sync.end / cur.end).toFixed(2)}×</b> faster. The best possible is the longest single stage,{' '}
            {Math.max(h2d, k, d2h)} ms. The CPU only enqueues work and is free until it synchronizes.
          </>
        )}
      </p>
      <p className="mt-2 font-mono text-[0.7rem] text-ink-faint">Assumes separate copy engines for each direction (true of most NVIDIA GPUs) and pinned host memory.</p>
    </div>
  );
}
