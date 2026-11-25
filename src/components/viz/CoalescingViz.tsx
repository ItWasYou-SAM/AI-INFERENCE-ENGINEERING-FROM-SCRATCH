import { useState } from 'react';

const SECTOR = 32; // bytes per DRAM sector the memory system fetches
const ELEM = 4; // float32

// Deterministic "random" gather indices.
const RAND = Array.from({ length: 32 }, (_, i) => ((i * 2654435761) >>> 7) % 512);

const PATTERNS = [
  { id: 'c', label: 'x[i]', addr: (l: number) => l, note: '32 adjacent floats = 128 bytes = 4 sectors. Every byte fetched is used.' },
  { id: 'm', label: 'x[i + 1]', addr: (l: number) => l + 1, note: 'Shifted by one element: the 128 bytes straddle 5 sectors. A small, usually acceptable cost.' },
  { id: 's2', label: 'x[2 * i]', addr: (l: number) => 2 * l, note: 'Every other float. Twice the sectors, half of each wasted.' },
  { id: 's8', label: 'x[8 * i]', addr: (l: number) => 8 * l, note: 'Each lane lands in its own sector. 32 sectors fetched to use 128 bytes: 1/8 efficiency.' },
  { id: 's32', label: 'x[i * N]  (a column)', addr: (l: number) => 32 * l, note: 'Reading down a column of a row-major matrix. Same 1/8 efficiency as stride 8, but spread across far more memory, which also hurts caches.' },
  { id: 'r', label: 'x[idx[i]]  (gather)', addr: (l: number) => RAND[l], note: 'Random gather, as in an embedding lookup by token id or a paged KV block table. Each lane is likely its own sector.' },
] as const;

export default function CoalescingViz() {
  const [pat, setPat] = useState<(typeof PATTERNS)[number]>(PATTERNS[0]);
  const addrs = Array.from({ length: 32 }, (_, l) => pat.addr(l) * ELEM);
  const sectors = Array.from(new Set(addrs.map((a) => Math.floor(a / SECTOR)))).sort((a, b) => a - b);
  const maxSector = Math.max(...sectors) + 1;
  const span = Math.max(16, maxSector);
  const fetched = sectors.length * SECTOR;
  const used = 32 * ELEM;
  const eff = used / fetched;

  const W = 640;
  const laneX = (l: number) => 20 + (l * (W - 40)) / 31;
  const secW = (W - 40) / span;
  const secX = (s: number) => 20 + s * secW;

  return (
    <div className="not-prose text-ink">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Lane i reads</span>
        {PATTERNS.map((p) => (
          <button key={p.id} type="button" aria-pressed={p === pat} onClick={() => setPat(p)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${p === pat ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {p.label}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} 190`} className="w-full" role="img" aria-label={`${sectors.length} sectors fetched, ${(eff * 100).toFixed(0)} percent of fetched bytes used`}>
        <text x={20} y={12} fontSize={10.5} className="fill-ink-soft font-mono">one warp: lanes 0–31</text>
        {addrs.map((a, l) => {
          const s = Math.floor(a / SECTOR);
          const within = (a % SECTOR) / SECTOR;
          return (
            <line key={l} x1={laneX(l)} y1={30} x2={secX(s) + (within + 0.0625) * secW} y2={130}
              stroke="var(--blue)" strokeWidth={1} opacity={0.5} />
          );
        })}
        {addrs.map((_, l) => (
          <circle key={l} cx={laneX(l)} cy={26} r={4.5} fill="var(--blue)" />
        ))}
        {Array.from({ length: span }, (_, s) => {
          const hit = sectors.includes(s);
          return (
            <rect key={s} x={secX(s) + 0.5} y={130} width={Math.max(1, secW - 1)} height={22} rx={1.5}
              fill={hit ? 'var(--amber)' : 'transparent'} opacity={hit ? 0.8 : 1}
              stroke="var(--rule)" strokeWidth={0.5} />
          );
        })}
        <text x={20} y={172} fontSize={10.5} className="fill-ink-soft font-mono">
          memory, in 32-byte sectors ({span} shown, {span * SECTOR} bytes){' '}
        </text>
        <text x={W - 20} y={172} textAnchor="end" fontSize={10.5} className="fill-amber font-mono">■ fetched</text>
      </svg>

      <dl className="mt-2 grid grid-cols-3 gap-3 font-mono text-xs">
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">sectors fetched</dt>
          <dd className="text-lg">{sectors.length}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">bytes fetched / used</dt>
          <dd className="text-lg">{fetched} / {used}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">efficiency</dt>
          <dd className="text-lg" style={{ color: eff > 0.75 ? 'var(--green)' : eff > 0.3 ? 'var(--amber)' : 'var(--red)' }}>{(eff * 100).toFixed(0)}%</dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-ink-soft">{pat.note}</p>
    </div>
  );
}
