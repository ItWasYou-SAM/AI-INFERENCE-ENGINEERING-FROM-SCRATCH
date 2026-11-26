import { useState } from 'react';

const BANKS = 32;

const RAND = Array.from({ length: 32 }, (_, i) => ((i * 2654435761) >>> 9) % 256);

const PATTERNS = [
  { id: 'b', label: 's[0]', addr: () => 0, note: 'Every lane reads the same word. The hardware reads it once and broadcasts it: one pass.' },
  { id: '1', label: 's[lane]', addr: (l: number) => l, note: 'Consecutive words land in consecutive banks: 32 lanes, 32 banks, one pass.' },
  { id: '2', label: 's[2 * lane]', addr: (l: number) => 2 * l, note: 'Only the even banks are used, and each holds two of the requested words: 2 passes.' },
  { id: '3', label: 's[3 * lane]', addr: (l: number) => 3 * l, note: 'An odd stride shares no factor with 32, so it still visits all 32 banks once: no conflict.' },
  { id: '8', label: 's[8 * lane]', addr: (l: number) => 8 * l, note: 'Only 4 banks are used, each holding 8 different words: 8 passes.' },
  { id: '32', label: 'tile[lane][c], tile[32][32]', addr: (l: number) => 32 * l, note: 'Reading down a column of a 32-wide float tile: every word is in the same bank. 32 passes, the worst case.' },
  { id: '33', label: 'tile[lane][c], tile[32][33]', addr: (l: number) => 33 * l, note: 'One column of padding shifts each row by one bank. The column now spans all 32 banks: one pass again.' },
  { id: 'r', label: 's[idx[lane]] (random)', addr: (l: number) => RAND[l], note: 'A data-dependent gather. A few banks get two or three requests by chance, so it takes a few passes.' },
] as const;

export default function BankConflicts() {
  const [pat, setPat] = useState<(typeof PATTERNS)[number]>(PATTERNS[5]);
  const addrs = Array.from({ length: 32 }, (_, l) => pat.addr(l));
  // per bank: the distinct words requested
  const perBank: number[][] = Array.from({ length: BANKS }, () => []);
  addrs.forEach((a) => { const b = a % BANKS; if (!perBank[b].includes(a)) perBank[b].push(a); });
  const passes = Math.max(...perBank.map((w) => w.length));
  const passOf = (a: number) => perBank[a % BANKS].indexOf(a);

  const W = 700;
  const lx = (l: number) => 20 + (l * (W - 40)) / 31;
  const bw = (W - 40) / BANKS;
  const bx = (b: number) => 20 + b * bw;
  const COLORS = ['var(--blue)', 'var(--amber)', 'var(--violet)', 'var(--teal)', 'var(--red)', 'var(--green)'];
  const cellH = Math.max(3, Math.min(12, 120 / passes));
  const baseY = 140;

  return (
    <div className="not-prose text-ink">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Lane reads</span>
        {PATTERNS.map((p) => (
          <button key={p.id} type="button" aria-pressed={p === pat} onClick={() => setPat(p)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${p === pat ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {p.label}
          </button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${baseY + passes * cellH + 46}`} className="w-full" role="img"
        aria-label={`${passes} pass${passes > 1 ? 'es' : ''}: the busiest bank holds ${passes} different requested words.`}>
        <text x={20} y={12} fontSize={10.5} className="fill-ink-soft font-mono">one warp: lanes 0–31</text>
        {addrs.map((a, l) => {
          const b = a % BANKS;
          return <line key={l} x1={lx(l)} y1={28} x2={bx(b) + bw / 2} y2={baseY - 4} stroke={COLORS[passOf(a) % COLORS.length]} strokeWidth={1.1} opacity={0.55} />;
        })}
        {addrs.map((a, l) => <circle key={l} cx={lx(l)} cy={24} r={4.5} fill={COLORS[passOf(a) % COLORS.length]} />)}
        {perBank.map((words, b) => (
          <g key={b}>
            <rect x={bx(b) + 1} y={baseY} width={bw - 2} height={Math.max(cellH, 6)} fill="transparent" stroke="var(--rule)" strokeWidth={0.6} />
            {words.map((w, k) => (
              <rect key={w} x={bx(b) + 1} y={baseY + k * cellH} width={bw - 2} height={cellH - 0.8} rx={1}
                fill={COLORS[k % COLORS.length]} opacity={0.8} />
            ))}
            {b % 4 === 0 && <text x={bx(b) + bw / 2} y={baseY - 8} fontSize={9} textAnchor="middle" className="fill-ink-faint font-mono">{b}</text>}
          </g>
        ))}
        <text x={20} y={baseY + passes * cellH + 22} fontSize={10.5} className="fill-ink-soft font-mono">
          32 banks; each stacked cell is one distinct word that bank must deliver; the colors are passes
        </text>
      </svg>

      <dl className="mt-2 grid grid-cols-3 gap-3 font-mono text-xs">
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">passes (n-way conflict)</dt>
          <dd className="text-lg" style={{ color: passes === 1 ? 'var(--green)' : passes <= 4 ? 'var(--amber)' : 'var(--red)' }}>{passes}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">banks used</dt>
          <dd className="text-lg">{perBank.filter((w) => w.length).length} / 32</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">shared-memory bandwidth</dt>
          <dd className="text-lg">{(100 / passes).toFixed(0)}%</dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-ink-soft">{pat.note}</p>
    </div>
  );
}
