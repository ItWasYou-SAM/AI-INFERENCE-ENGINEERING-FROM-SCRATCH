import { useMemo, useState } from 'react';

/** Same roofline cost model as labs/m03/3.5/derive_ai_7b.py, BF16 weights and KV cache. */
const MODELS = [
  { name: 'Llama-2-7B', L: 32, D: 4096, KV: 32, DH: 128, FF: 11008, V: 32000 },
  { name: 'Llama-3-8B', L: 32, D: 4096, KV: 8, DH: 128, FF: 14336, V: 128256 },
];
const GPUS = [
  { name: 'A100 80GB', peak: 312e12, bw: 2.04e12 },
  { name: 'H100 SXM', peak: 989e12, bw: 3.35e12 },
  { name: 'H200', peak: 989e12, bw: 4.8e12 },
];
type Model = (typeof MODELS)[number];
type Gpu = (typeof GPUS)[number];
const B = 2;

/** Roofline time of one forward step: sum over ops of max(FLOPs / peak, bytes / bandwidth). */
function stepTime(m: Model, g: Gpu, batch: number, newTok: number, ctx: number) {
  const M = batch * newTok;
  const kvd = m.KV * m.DH;
  const mm = (a: number, k: number, n: number): [number, number] => [2 * a * k * n, B * (a * k + k * n + a * n)];
  const ops: [number, number][] = [mm(M, m.D, m.D + 2 * kvd), mm(M, m.D, m.D), mm(M, m.D, 2 * m.FF), mm(M, m.FF, m.D)];
  const attn: [number, number] =
    newTok === 1
      ? [batch * 4 * ctx * m.D, batch * (2 * ctx * kvd * B + 2 * m.D * B)]
      : [(batch * 4 * newTok * newTok * m.D) / 2, batch * B * (2 * newTok * m.D + 2 * newTok * kvd)];
  ops.push(attn);
  let t = 0;
  let tCompute = 0;
  for (const [f, b] of ops) {
    const tc = f / g.peak;
    const tm = b / g.bw;
    t += m.L * Math.max(tc, tm);
    if (tc > tm) tCompute += m.L * tc;
  }
  const [hf, hb] = mm(batch, m.D, m.V);
  t += Math.max(hf / g.peak, hb / g.bw);
  return { t, computeShare: tCompute / t };
}

const fmtMs = (s: number) => (s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1e3).toFixed(s < 0.01 ? 2 : 1)} ms`);

export default function PrefillDecodeTimeline() {
  const [model, setModel] = useState(MODELS[1]);
  const [gpu, setGpu] = useState(GPUS[1]);
  const [prompt, setPrompt] = useState(2048);
  const [out, setOut] = useState(256);
  const [batch, setBatch] = useState(1);

  const r = useMemo(() => {
    const pre = stepTime(model, gpu, 1, prompt, prompt);
    let decodeTotal = 0;
    let computeShare = 0;
    const stride = Math.max(1, Math.floor(out / 64)); // sample steps for speed, then scale
    let samples = 0;
    for (let i = 1; i < out; i += stride) {
      const s = stepTime(model, gpu, batch, 1, prompt + i);
      decodeTotal += s.t * Math.min(stride, out - i);
      computeShare += s.computeShare;
      samples++;
    }
    const tpot = out > 1 ? decodeTotal / (out - 1) : 0;
    return {
      ttft: pre.t,
      preCompute: pre.computeShare,
      tpot,
      decodeTotal,
      decCompute: samples ? computeShare / samples : 0,
      e2e: pre.t + decodeTotal,
    };
  }, [model, gpu, prompt, out, batch]);

  const preFrac = r.ttft / r.e2e;
  const sliders = [
    { label: 'prompt tokens', v: prompt, set: setPrompt, min: 6, max: 15, color: 'var(--blue)' },
    { label: 'output tokens', v: out, set: setOut, min: 4, max: 12, color: 'var(--red)' },
    { label: 'concurrent users decoding', v: batch, set: setBatch, min: 0, max: 8, color: 'var(--violet)' },
  ];

  return (
    <div className="not-prose text-ink">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {MODELS.map((m) => (
          <button key={m.name} type="button" aria-pressed={m === model} onClick={() => setModel(m)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${m === model ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {m.name}
          </button>
        ))}
        <span className="mx-1 text-ink-faint">·</span>
        {GPUS.map((g) => (
          <button key={g.name} type="button" aria-pressed={g === gpu} onClick={() => setGpu(g)}
            className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${g === gpu ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {g.name}
          </button>
        ))}
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {sliders.map((s) => (
          <label key={s.label} className="block">
            <span className="eyebrow flex justify-between" style={{ color: s.color }}>
              {s.label} <b>{s.v.toLocaleString()}</b>
            </span>
            <input type="range" min={s.min} max={s.max} step={1} value={Math.log2(s.v)} onChange={(e) => s.set(2 ** +e.target.value)}
              className="mt-1 w-full" style={{ accentColor: s.color }} />
          </label>
        ))}
      </div>

      {/* timeline */}
      <div className="mt-6">
        <div className="flex h-12 overflow-hidden rounded-md border border-rule" role="img"
          aria-label={`Prefill ${fmtMs(r.ttft)}, then ${out - 1} decode steps of ${fmtMs(r.tpot)} each`}>
          <div className="flex items-center justify-center font-mono text-[0.7rem] text-paper"
            style={{ width: `${Math.max(preFrac * 100, 0.6)}%`, background: 'var(--blue)', transition: 'width 0.3s ease' }}>
            {preFrac > 0.12 ? 'prefill' : ''}
          </div>
          <div className="relative flex-1" style={{ background: 'color-mix(in srgb, var(--red) 14%, transparent)' }}>
            <div className="absolute inset-0" style={{
              backgroundImage: `repeating-linear-gradient(90deg, var(--red) 0 1px, transparent 1px ${Math.max(2, 100 / Math.min(out, 200))}%)`,
              opacity: 0.55,
            }} />
            <span className="absolute inset-0 flex items-center justify-center font-mono text-[0.7rem] text-ink">
              decode: {out - 1} steps, one token each
            </span>
          </div>
        </div>
        <div className="mt-1 flex justify-between font-mono text-[0.65rem] text-ink-faint">
          <span>request arrives</span>
          <span>↑ first token at {fmtMs(r.ttft)}</span>
          <span>last token at {fmtMs(r.e2e)}</span>
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-3 font-mono text-xs sm:grid-cols-4">
        {[
          { k: 'TTFT', v: fmtMs(r.ttft), c: 'var(--blue)' },
          { k: 'TPOT', v: fmtMs(r.tpot), c: 'var(--red)' },
          { k: 'end to end', v: fmtMs(r.e2e), c: 'var(--ink)' },
          { k: 'tokens/s per user', v: r.tpot ? (1 / r.tpot).toFixed(0) : '—', c: 'var(--ink)' },
        ].map((s) => (
          <div key={s.k} className="rounded-md border border-rule bg-paper p-3">
            <dt className="text-ink-faint">{s.k}</dt>
            <dd className="text-lg" style={{ color: s.c }}>{s.v}</dd>
          </div>
        ))}
      </dl>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <p className="rounded-md border border-rule p-3 text-sm text-ink-soft">
          <b style={{ color: 'var(--blue)' }}>Prefill</b> is{' '}
          <b className="text-ink">{r.preCompute > 0.5 ? 'compute-bound' : 'memory-bound'}</b> ({(r.preCompute * 100).toFixed(0)}% of its time in
          compute-limited ops). It processes {prompt.toLocaleString()} tokens in one pass.
        </p>
        <p className="rounded-md border border-rule p-3 text-sm text-ink-soft">
          <b style={{ color: 'var(--red)' }}>Decode</b> is{' '}
          <b className="text-ink">{r.decCompute > 0.5 ? 'compute-bound' : 'memory-bound'}</b>. Each step makes {batch} token{batch > 1 ? 's' : ''}, so the GPU
          produces <b className="text-ink">{r.tpot ? Math.round(batch / r.tpot).toLocaleString() : '—'}</b> tokens/s in total.
        </p>
      </div>
      <p className="mt-3 font-mono text-[0.7rem] text-ink-faint">
        Roofline lower bounds in BF16, prefill of this one request alone. Real engines reach maybe 60–85% of these speeds.
      </p>
    </div>
  );
}
