import { useState } from 'react';

const GPUS = [
  { name: 'RTX 3050 Laptop', sms: 20 },
  { name: 'A100', sms: 108 },
  { name: 'H100 SXM', sms: 132 },
] as const;
const TILES = [64, 128, 256] as const;

export default function WaveQuantization() {
  const [gpu, setGpu] = useState<(typeof GPUS)[number]>(GPUS[2]);
  const [perSM, setPerSM] = useState(1);
  const [tileSize, setTileSize] = useState<(typeof TILES)[number]>(128);
  const [M, setM] = useState(4096);
  const [Nn, setN] = useState(4352);

  const tilesM = Math.ceil(M / tileSize), tilesN = Math.ceil(Nn / tileSize);
  const tiles = tilesM * tilesN;
  const slots = gpu.sms * perSM;
  const waves = Math.ceil(tiles / slots);
  const last = tiles - (waves - 1) * slots;
  const util = tiles / (waves * slots);
  const ideal = tiles / slots; // waves if the work could be split perfectly

  const W = 680;
  const rowH = 26;
  const maxRows = Math.min(waves, 12);
  const showCells = slots <= 64;

  return (
    <div className="not-prose text-ink">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">GPU</span>
        {GPUS.map((g) => (
          <button key={g.name} type="button" aria-pressed={g === gpu} onClick={() => setGpu(g)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${g === gpu ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {g.name} · {g.sms} SMs
          </button>
        ))}
      </div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Blocks per SM</span>
        {[1, 2, 4].map((b) => (
          <button key={b} type="button" aria-pressed={b === perSM} onClick={() => setPerSM(b)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${b === perSM ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {b}
          </button>
        ))}
        <span className="eyebrow ml-3 mr-1">Tile</span>
        {TILES.map((t) => (
          <button key={t} type="button" aria-pressed={t === tileSize} onClick={() => setTileSize(t)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${t === tileSize ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {t}×{t}
          </button>
        ))}
      </div>
      <div className="mb-4 grid gap-4 sm:grid-cols-2">
        {[
          { label: 'M (rows of C)', v: M, set: setM },
          { label: 'N (columns of C)', v: Nn, set: setN },
        ].map((s) => (
          <label key={s.label} className="block">
            <span className="eyebrow flex justify-between">{s.label} <b className="text-ink">{s.v}</b></span>
            <input type="range" min={128} max={8192} step={128} value={s.v} onChange={(e) => s.set(+e.target.value)} className="mt-1 w-full accent-[var(--blue)]" />
          </label>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${maxRows * rowH + 40}`} className="w-full" role="img"
        aria-label={`${tiles} tiles on ${slots} slots: ${waves} waves, the last one ${Math.round((100 * last) / slots)} percent full.`}>
        <text x={0} y={12} fontSize={10.5} className="fill-ink-soft font-mono">
          each row is one wave: {slots} block slots ({gpu.sms} SMs × {perSM})
        </text>
        {Array.from({ length: maxRows }, (_, w) => {
          const isLast = w === waves - 1 || (w === maxRows - 1 && waves > maxRows);
          const filled = w === waves - 1 ? last : slots;
          const y = 22 + w * rowH;
          const x0 = 70, bw = W - x0 - 10;
          return (
            <g key={w}>
              <text x={0} y={y + 16} fontSize={10.5} className="fill-ink-faint font-mono">
                {w === maxRows - 1 && waves > maxRows ? `… ${waves}` : `wave ${w + 1}`}
              </text>
              <rect x={x0} y={y} width={bw} height={rowH - 6} rx={3} fill="url(#hatch)" stroke="var(--rule)" />
              {showCells
                ? Array.from({ length: filled }, (_, i) => (
                    <rect key={i} x={x0 + (i * bw) / slots + 0.6} y={y + 0.6} width={bw / slots - 1.2} height={rowH - 7.2} rx={1.5}
                      fill={w === waves - 1 && last < slots ? 'var(--amber)' : 'var(--blue)'} opacity={0.8} />
                  ))
                : <rect x={x0} y={y} width={(filled / slots) * bw} height={rowH - 6} rx={3}
                    fill={w === waves - 1 && last < slots ? 'var(--amber)' : 'var(--blue)'} opacity={0.8} />}
              {isLast && w === waves - 1 && last < slots && (
                <text x={x0 + (filled / slots) * bw + 6} y={y + 15} fontSize={10.5} className="fill-ink-soft font-mono">
                  {last} of {slots} busy
                </text>
              )}
            </g>
          );
        })}
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--rule)" strokeWidth="1.5" />
          </pattern>
        </defs>
      </svg>

      <dl className="mt-2 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">tiles</dt>
          <dd className="text-lg">{tilesM} × {tilesN} = {tiles}</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">waves</dt>
          <dd className="text-lg">{waves} <span className="text-xs text-ink-faint">(ideal {ideal.toFixed(2)})</span></dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">SM utilization</dt>
          <dd className="text-lg" style={{ color: util > 0.9 ? 'var(--green)' : util > 0.7 ? 'var(--amber)' : 'var(--red)' }}>{(util * 100).toFixed(0)}%</dd>
        </div>
        <div className="rounded-md border border-rule bg-paper p-3">
          <dt className="text-ink-faint">time vs perfect split</dt>
          <dd className="text-lg">{(waves / ideal).toFixed(2)}×</dd>
        </div>
      </dl>
      <p className="mt-3 text-sm text-ink-soft">
        Every wave takes as long as one block, however few blocks it holds. Here {tiles} tiles need {waves} waves, and the last one keeps only {last} of {slots} slots busy.
        {' '}On an H100 with M = 4096, try N = 4224 (exactly 8 full waves), then 4352: one more column of tiles adds a ninth wave that is three-quarters empty.
      </p>
    </div>
  );
}
