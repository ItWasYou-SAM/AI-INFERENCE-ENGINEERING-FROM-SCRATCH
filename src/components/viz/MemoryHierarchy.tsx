import { useState } from 'react';

/**
 * H100 SXM, approximate. Capacities and HBM/PCIe bandwidth are from NVIDIA's datasheet and whitepaper.
 * Latencies and on-chip bandwidths are typical microbenchmark values and vary by access pattern.
 */
const CLOCK_GHZ = 1.98;
const LEVELS = [
  { name: 'Registers', where: 'inside each SM', cap: 256 * 1024 * 132, capNote: '256 KB × 132 SMs', bw: null as number | null, lat: 1, color: 'var(--green)' },
  { name: 'Shared mem / L1', where: 'inside each SM', cap: 256 * 1024 * 132, capNote: '256 KB × 132 SMs', bw: 33e12, lat: 30, color: 'var(--teal)' },
  { name: 'L2 cache', where: 'on the GPU die', cap: 50 * 2 ** 20, capNote: '50 MB, shared', bw: 10e12, lat: 250, color: 'var(--blue)' },
  { name: 'HBM3', where: 'stacked next to the die', cap: 80e9, capNote: '80 GB', bw: 3.35e12, lat: 600, color: 'var(--violet)' },
  { name: 'Host DRAM', where: 'over PCIe Gen5', cap: 2e12, capNote: 'up to TBs', bw: 64e9, lat: 3000, color: 'var(--red)' },
];

const PRESETS = [
  { label: 'attention tile', bytes: 64 * 1024 },
  { label: 'one 8B layer', bytes: 218e6 * 2 },
  { label: '4K-token KV (8B)', bytes: 4096 * 128 * 1024 },
  { label: 'all 8B weights', bytes: 16e9 },
  { label: '70B weights', bytes: 140e9 },
];

const fmtBytes = (b: number) =>
  b >= 1e12 ? `${(b / 1e12).toFixed(1)} TB` : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`;
const fmtTime = (s: number) =>
  s >= 1 ? `${s.toFixed(2)} s` : s >= 1e-3 ? `${(s * 1e3).toFixed(2)} ms` : s >= 1e-6 ? `${(s * 1e6).toFixed(2)} µs` : `${(s * 1e9).toFixed(1)} ns`;
const fmtHuman = (cycles: number) => {
  // 1 cycle = 1 second
  if (cycles < 60) return `${cycles} second${cycles === 1 ? "" : "s"}`;
  if (cycles < 3600) return `${(cycles / 60).toFixed(0)} min`;
  return `${(cycles / 3600).toFixed(1)} h`;
};

const LOG_MIN = 3; // 1 KB
const LOG_MAX = 12.3; // 2 TB

export default function MemoryHierarchy() {
  const [logB, setLogB] = useState(Math.log10(16e9));
  const [human, setHuman] = useState(false);
  const bytes = 10 ** logB;
  const home = LEVELS.findIndex((l) => l.cap >= bytes);

  return (
    <div className="not-prose text-ink">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">Data</span>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            aria-pressed={Math.abs(Math.log10(p.bytes) - logB) < 0.01}
            onClick={() => setLogB(Math.log10(p.bytes))}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${
              Math.abs(Math.log10(p.bytes) - logB) < 0.01 ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <label className="mb-5 block">
        <span className="eyebrow flex justify-between">
          working set <b className="text-ink">{fmtBytes(bytes)}</b>
        </span>
        <input type="range" min={LOG_MIN} max={LOG_MAX} step={0.01} value={logB} onChange={(e) => setLogB(+e.target.value)} className="mt-2 w-full" />
      </label>

      <div className="space-y-2">
        {LEVELS.map((l, i) => {
          const fits = l.cap >= bytes;
          const isHome = i === home;
          const t = l.bw ? bytes / l.bw : null;
          return (
            <div
              key={l.name}
              className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)] items-center gap-3 rounded-md border px-3 py-2 text-sm transition-colors"
              style={{
                borderColor: isHome ? l.color : 'var(--rule)',
                background: isHome ? `color-mix(in srgb, ${l.color} 10%, transparent)` : 'transparent',
                opacity: fits ? 1 : 0.5,
              }}
            >
              <div>
                <p className="font-serif" style={{ color: l.color }}>{l.name}</p>
                <p className="font-mono text-[0.65rem] text-ink-faint">{l.where}</p>
              </div>
              <div className="font-mono text-xs">
                <p className="text-ink-faint">capacity</p>
                <p>{l.capNote}</p>
              </div>
              <div className="font-mono text-xs">
                <p className="text-ink-faint">latency</p>
                <p>{human ? fmtHuman(l.lat) : `~${l.lat} cycle${l.lat === 1 ? "" : "s"}`}</p>
              </div>
              <div className="font-mono text-xs">
                <p className="text-ink-faint">read it once</p>
                <p className={fits ? 'text-ink' : 'text-ink-faint'}>
                  {!fits ? "doesn't fit" : t === null ? 'every cycle' : fmtTime(t)}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-ink-soft">
          {home === -1 ? (
            <>Too big for one GPU and its host. Split it across machines.</>
          ) : (
            <>
              {fmtBytes(bytes)} fits in <b style={{ color: LEVELS[home].color }}>{LEVELS[home].name}</b>
              {home >= 3 && LEVELS[home].bw && (
                <>
                  . Streaming it from there takes <b className="text-ink">{fmtTime(bytes / LEVELS[home].bw!)}</b>, during which the GPU could do{' '}
                  <b className="text-ink">{((bytes / LEVELS[home].bw!) * 989).toFixed(bytes / LEVELS[home].bw! < 0.01 ? 2 : 0)} TFLOP</b> of BF16 math
                </>
              )}
              .
            </>
          )}
        </p>
        <label className="flex items-center gap-2 font-mono text-xs text-ink-soft">
          <input type="checkbox" checked={human} onChange={(e) => setHuman(e.target.checked)} style={{ accentColor: 'var(--amber)' }} />
          human scale (1 cycle = 1 second)
        </label>
      </div>
      <p className="mt-3 font-mono text-[0.7rem] text-ink-faint">
        H100 SXM at {CLOCK_GHZ} GHz. Latencies and on-chip bandwidths are approximate microbenchmark figures.
      </p>
    </div>
  );
}
