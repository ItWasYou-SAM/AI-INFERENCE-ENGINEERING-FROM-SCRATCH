import { useMemo, useRef, useState } from 'react';
import { motion, MotionConfig } from 'motion/react';

/** Dense BF16/FP16 tensor-core peak (TFLOP/s) and HBM/GDDR bandwidth (TB/s), from vendor datasheets. */
const GPUS = [
  { name: 'L4', peak: 121, bw: 0.3 },
  { name: 'RTX 4090', peak: 165, bw: 1.01 },
  { name: 'A100 80GB', peak: 312, bw: 2.04 },
  { name: 'H100 SXM', peak: 989, bw: 3.35 },
];

// One d×d linear layer in FP16 (2 bytes): X[M×d] · W[d×d]. M = tokens processed in this step.
const D = 4096;
const flops = (M: number) => 2 * M * D * D;
const bytes = (M: number) => 2 * (M * D + D * D + M * D); // read X, read W, write Y
const intensity = (M: number) => flops(M) / bytes(M);

// Plot geometry
const W = 680;
const H = 400;
const PAD = { l: 64, r: 20, t: 20, b: 52 };
const X_MIN = 0.1;
const X_MAX = 10000;
const Y_MIN = 0.1;
const Y_MAX = 2000;
const lx = (v: number) => PAD.l + ((Math.log10(v) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (W - PAD.l - PAD.r);
const ly = (v: number) => H - PAD.b - ((Math.log10(v) - Math.log10(Y_MIN)) / (Math.log10(Y_MAX) - Math.log10(Y_MIN))) * (H - PAD.t - PAD.b);
const invX = (px: number) => 10 ** (Math.log10(X_MIN) + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (Math.log10(X_MAX) - Math.log10(X_MIN)));

/** Smallest M whose intensity reaches target (intensity is monotonic in M). */
function mForIntensity(target: number, lo: number, hi: number) {
  let a = lo;
  let b = hi;
  if (intensity(b) <= target) return b;
  for (let i = 0; i < 40; i++) {
    const mid = (a + b) / 2;
    if (intensity(mid) < target) a = mid;
    else b = mid;
  }
  return Math.max(lo, Math.round(b));
}

const fmtT = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));

export default function RooflinePlot() {
  const [gpu, setGpu] = useState(GPUS[3]);
  const [batch, setBatch] = useState(8); // decode: M = batch (one new token per sequence)
  const [prefill, setPrefill] = useState(2048); // prefill: M = prompt tokens
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<null | 'decode' | 'prefill'>(null);

  const ridge = gpu.peak / gpu.bw; // FLOP/byte (TFLOP/s ÷ TB/s)
  const attain = (ai: number) => Math.min(gpu.peak, gpu.bw * ai);

  const pts = useMemo(
    () =>
      [
        { id: 'decode' as const, label: `decode · batch ${batch}`, M: batch, color: 'var(--red)' },
        { id: 'prefill' as const, label: `prefill · ${prefill} tokens`, M: prefill, color: 'var(--blue)' },
      ].map((p) => {
        const ai = intensity(p.M);
        const perf = attain(ai);
        return { ...p, ai, perf, util: perf / gpu.peak, bound: ai < ridge ? 'memory-bound' : 'compute-bound' };
      }),
    [batch, prefill, gpu],
  );

  const onMove = (e: React.PointerEvent) => {
    if (!drag || !svgRef.current) return;
    const r = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const ai = invX(Math.min(W - PAD.r, Math.max(PAD.l, px)));
    if (drag === 'decode') setBatch(mForIntensity(ai, 1, 1024));
    else setPrefill(mForIntensity(ai, 16, 32768));
  };

  const roof = `M ${lx(X_MIN)} ${ly(attain(X_MIN))} L ${lx(ridge)} ${ly(gpu.peak)} L ${lx(X_MAX)} ${ly(gpu.peak)}`;
  const xTicks = [0.1, 1, 10, 100, 1000, 10000];
  const yTicks = [0.1, 1, 10, 100, 1000];

  return (
    <MotionConfig reducedMotion="user">
    <div className="not-prose text-ink">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="eyebrow mr-1">GPU</span>
        {GPUS.map((g) => (
          <button
            key={g.name}
            type="button"
            aria-pressed={g === gpu}
            onClick={() => setGpu(g)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${
              g === gpu ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'
            }`}
          >
            {g.name}
          </button>
        ))}
        <span className="ml-auto font-mono text-[0.7rem] text-ink-faint">
          {gpu.peak} TFLOP/s · {gpu.bw} TB/s · ridge ≈ {ridge.toFixed(0)} FLOP/B
        </span>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-none select-none"
        onPointerMove={onMove}
        onPointerUp={() => setDrag(null)}
        onPointerLeave={() => setDrag(null)}
        role="img"
        aria-label={`Roofline for ${gpu.name}. Decode at batch ${batch} is ${pts[0].bound}; prefill of ${prefill} tokens is ${pts[1].bound}.`}
      >
        {/* regions */}
        <rect x={PAD.l} y={PAD.t} width={lx(ridge) - PAD.l} height={H - PAD.t - PAD.b} fill="var(--red)" opacity={0.05} />
        <rect x={lx(ridge)} y={PAD.t} width={W - PAD.r - lx(ridge)} height={H - PAD.t - PAD.b} fill="var(--blue)" opacity={0.05} />
        <text x={PAD.l + 10} y={PAD.t + 18} className="fill-red font-mono" fontSize={11} opacity={0.8}>
          MEMORY-BOUND
        </text>
        <text x={W - PAD.r - 10} y={PAD.t + 18} textAnchor="end" className="fill-blue font-mono" fontSize={11} opacity={0.8}>
          COMPUTE-BOUND
        </text>

        {/* grid + axes */}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={lx(t)} x2={lx(t)} y1={PAD.t} y2={H - PAD.b} stroke="var(--rule)" strokeDasharray="2 4" />
            <text x={lx(t)} y={H - PAD.b + 18} textAnchor="middle" fontSize={11} className="fill-ink-soft font-mono">
              {t}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={ly(t)} y2={ly(t)} stroke="var(--rule)" strokeDasharray="2 4" />
            <text x={PAD.l - 8} y={ly(t) + 4} textAnchor="end" fontSize={11} className="fill-ink-soft font-mono">
              {t}
            </text>
          </g>
        ))}
        <text x={(W + PAD.l) / 2} y={H - 10} textAnchor="middle" fontSize={12} className="fill-ink font-serif italic">
          arithmetic intensity (FLOP / byte), log scale
        </text>
        <text transform={`translate(16 ${(H - PAD.b + PAD.t) / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} className="fill-ink font-serif italic">
          attainable TFLOP/s, log scale
        </text>

        {/* the roof: redraws on GPU change */}
        <motion.path
          key={gpu.name}
          d={roof}
          fill="none"
          stroke="var(--ink)"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.9, ease: 'easeInOut' }}
        />
        <line x1={lx(ridge)} x2={lx(ridge)} y1={ly(gpu.peak)} y2={H - PAD.b} stroke="var(--ink-faint)" strokeDasharray="4 4" />
        <text x={lx(ridge) + 6} y={H - PAD.b - 8} fontSize={10.5} className="fill-ink-soft font-mono">
          ridge
        </text>

        {/* operating points */}
        {pts.map((p) => (
          <g key={p.id}>
            <motion.line
              animate={{ x1: lx(p.ai), x2: lx(p.ai), y1: ly(p.perf), y2: H - PAD.b }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              stroke={p.color}
              strokeDasharray="3 3"
              opacity={0.6}
            />
            <motion.g
              animate={{ x: lx(p.ai), y: ly(p.perf) }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              style={{ cursor: drag === p.id ? 'grabbing' : 'grab' }}
              onPointerDown={(e) => {
                (e.target as Element).setPointerCapture?.(e.pointerId);
                setDrag(p.id);
              }}
            >
              <circle r={16} fill="transparent" />
              <circle r={7} fill={p.color} stroke="var(--paper)" strokeWidth={2} />
              <text y={p.id === 'decode' ? -14 : 22} textAnchor="middle" fontSize={11.5} className="font-mono" fill={p.color}>
                {p.label}
              </text>
            </motion.g>
          </g>
        ))}
      </svg>

      <div className="mt-4 grid gap-5 sm:grid-cols-2">
        {pts.map((p) => (
          <div key={p.id} className="rounded-md border border-rule bg-paper p-4">
            <label className="block">
              <span className="eyebrow flex justify-between" style={{ color: p.color }}>
                {p.id === 'decode' ? 'Decode batch size' : 'Prefill tokens'}
                <b>{p.M.toLocaleString()}</b>
              </span>
              {p.id === 'decode' ? (
                <input type="range" min={0} max={10} step={1} value={Math.log2(batch)} onChange={(e) => setBatch(2 ** +e.target.value)} className="mt-2 w-full" style={{ accentColor: p.color }} />
              ) : (
                <input type="range" min={4} max={15} step={1} value={Math.log2(prefill)} onChange={(e) => setPrefill(2 ** +e.target.value)} className="mt-2 w-full" style={{ accentColor: p.color }} />
              )}
            </label>
            <dl className="mt-3 grid grid-cols-3 gap-2 font-mono text-xs">
              <div>
                <dt className="text-ink-faint">intensity</dt>
                <dd className="text-base text-ink">{fmtT(p.ai)}</dd>
              </div>
              <div>
                <dt className="text-ink-faint">TFLOP/s</dt>
                <dd className="text-base text-ink">{fmtT(p.perf)}</dd>
              </div>
              <div>
                <dt className="text-ink-faint">of peak</dt>
                <dd className="text-base" style={{ color: p.bound === 'memory-bound' ? 'var(--red)' : 'var(--green)' }}>
                  {(p.util * 100).toFixed(p.util < 0.1 ? 1 : 0)}%
                </dd>
              </div>
            </dl>
            <p className="mt-2 text-sm text-ink-soft">
              <b style={{ color: p.bound === 'memory-bound' ? 'var(--red)' : 'var(--green)' }}>{p.bound}</b>
              {p.bound === 'memory-bound' ? ' — the tensor cores wait on HBM.' : ' — HBM keeps up; math is the limit.'}
            </p>
          </div>
        ))}
      </div>
      <p className="mt-3 font-mono text-[0.7rem] text-ink-faint">
        Model: one {D}×{D} FP16 linear layer. Drag the dots or use the sliders. Real kernels sit below the roof.
      </p>
    </div>
    </MotionConfig>
  );
}
