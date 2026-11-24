import { useMemo, useState } from 'react';
import { animate, motion, MotionConfig, useMotionValue, useTransform } from 'motion/react';
import { useEffect } from 'react';

/** Datasheet numbers: dense BF16 tensor TFLOP/s, memory TB/s, memory GB. */
const GPUS = [
  { name: 'RTX 4090', tflops: 165, bw: 1.01, mem: 24 },
  { name: 'L4', tflops: 121, bw: 0.3, mem: 24 },
  { name: 'A100 80GB', tflops: 312, bw: 2.04, mem: 80 },
  { name: 'H100 SXM', tflops: 989, bw: 3.35, mem: 80 },
  { name: 'H200', tflops: 989, bw: 4.8, mem: 141 },
  { name: 'B200', tflops: 2250, bw: 8.0, mem: 180 },
];
const MODELS = [
  { name: '1B', params: 1.2 },
  { name: '8B', params: 8.0 },
  { name: '32B', params: 32.8 },
  { name: '70B', params: 70.6 },
];
const PREC = [
  { name: 'FP16', bytes: 2 },
  { name: 'FP8', bytes: 1 },
  { name: 'INT4', bytes: 0.5 },
];
const BATCHES = [1, 4, 16, 64, 256];

function Counter({ value, digits = 0 }: { value: number; digits?: number }) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }));
  useEffect(() => {
    const c = animate(mv, value, { duration: 0.6, ease: 'easeOut' });
    return c.stop;
  }, [value, mv]);
  return <motion.span>{text}</motion.span>;
}

function Pills<T extends { name: string }>({ items, value, onChange, id }: { items: T[]; value: T; onChange: (v: T) => void; id: string }) {
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((it) => (
        <button
          key={it.name}
          type="button"
          aria-pressed={it === value}
          onClick={() => onChange(it)}
          className={`relative rounded-full px-3 py-1 font-mono text-xs transition-colors ${it === value ? 'text-paper' : 'text-ink-soft hover:text-ink'}`}
        >
          {it === value && (
            <motion.span layoutId={`sol-${id}`} className="absolute inset-0 rounded-full" style={{ background: 'var(--ink)' }} transition={{ type: 'spring', stiffness: 500, damping: 36 }} />
          )}
          <span className="relative">{it.name}</span>
        </button>
      ))}
    </div>
  );
}

export default function SpeedOfLight() {
  const [gpu, setGpu] = useState(GPUS[3]);
  const [model, setModel] = useState(MODELS[1]);
  const [prec, setPrec] = useState(PREC[0]);
  const [bi, setBi] = useState(0);
  const batch = BATCHES[bi];

  const r = useMemo(() => {
    const weightBytes = model.params * 1e9 * prec.bytes;
    const memMs = (weightBytes / (gpu.bw * 1e12)) * 1e3; // read every weight once per step
    const flops = 2 * model.params * 1e9 * batch;
    const compMs = (flops / (gpu.tflops * 1e12)) * 1e3;
    const stepMs = Math.max(memMs, compMs);
    return {
      weightsGB: weightBytes / 1e9,
      fits: weightBytes / 1e9 < gpu.mem * 0.9,
      memMs,
      compMs,
      stepMs,
      perUser: 1000 / stepMs,
      total: (1000 / stepMs) * batch,
      bound: memMs >= compMs ? 'memory' : 'compute',
    };
  }, [gpu, model, prec, batch]);

  const maxMs = Math.max(r.memMs, r.compMs);

  return (
    <MotionConfig reducedMotion="user">
      <div className="sheet relative overflow-hidden rounded-2xl p-5 sm:p-7">
        <div className="grid gap-7 lg:grid-cols-[1fr_1.1fr]">
          <div className="space-y-4">
            <div>
              <p className="eyebrow mb-1.5">Model size</p>
              <Pills id="m" items={MODELS} value={model} onChange={setModel} />
            </div>
            <div>
              <p className="eyebrow mb-1.5">Weight precision</p>
              <Pills id="p" items={PREC} value={prec} onChange={setPrec} />
            </div>
            <div>
              <p className="eyebrow mb-1.5">GPU</p>
              <Pills id="g" items={GPUS} value={gpu} onChange={setGpu} />
            </div>
            <label className="block">
              <span className="eyebrow flex justify-between">
                Users decoding together (batch) <b className="text-ink">{batch}</b>
              </span>
              <input type="range" min={0} max={BATCHES.length - 1} value={bi} onChange={(e) => setBi(+e.target.value)} className="mt-2 w-full accent-[var(--blue)]" />
            </label>
          </div>

          <div className="flex flex-col">
            {r.fits ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-paper-2 p-4">
                    <p className="eyebrow">Per user</p>
                    <p className="mt-1 font-display text-4xl tabular-nums">
                      <span className="text-blue">
                        <Counter value={r.perUser} />
                      </span>
                      <span className="ml-1 text-base text-ink-soft">tok/s</span>
                    </p>
                  </div>
                  <div className="rounded-xl bg-paper-2 p-4">
                    <p className="eyebrow">Whole GPU</p>
                    <p className="mt-1 font-display text-4xl tabular-nums">
                      <Counter value={r.total} />
                      <span className="ml-1 text-base text-ink-soft">tok/s</span>
                    </p>
                  </div>
                </div>

                <p className="eyebrow mt-5 mb-2">Time for one decode step</p>
                {[
                  { k: 'Moving weights from memory', ms: r.memMs, c: 'var(--ink-soft)' },
                  { k: 'Doing the math', ms: r.compMs, c: 'var(--blue)' },
                ].map((row) => (
                  <div key={row.k} className="mb-2.5">
                    <div className="mb-1 flex justify-between font-mono text-[0.7rem] text-ink-soft">
                      <span>{row.k}</span>
                      <span>{row.ms < 0.1 ? row.ms.toFixed(3) : row.ms.toFixed(2)} ms</span>
                    </div>
                    <div className="h-2.5 overflow-hidden rounded-full bg-paper-3">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ background: row.c }}
                        animate={{ width: `${Math.max(0.8, (row.ms / maxMs) * 100)}%` }}
                        transition={{ type: 'spring', stiffness: 120, damping: 20 }}
                      />
                    </div>
                  </div>
                ))}
                <motion.p key={r.bound} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mt-2 text-[0.95rem] leading-snug">
                  {r.bound === 'memory' ? (
                    <>
                      <b>Memory-bound.</b> The cores wait {((1 - r.compMs / r.memMs) * 100).toFixed(1)}% of every step for weights. Add
                      users: they ride along almost for free.
                    </>
                  ) : (
                    <>
                      <b className="text-blue">Compute-bound.</b> The batch is big enough that the math, not memory, sets the pace. More users now
                      cost real time.
                    </>
                  )}
                </motion.p>
              </>
            ) : (
              <motion.div initial={{ scale: 0.96, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="grid flex-1 place-items-center rounded-xl border border-dashed border-red p-6 text-center">
                <div>
                  <p className="font-display text-2xl text-red">Doesn’t fit.</p>
                  <p className="mt-2 text-sm text-ink-soft">
                    {model.name} in {prec.name} needs {r.weightsGB.toFixed(0)} GB of weights. The {gpu.name} has {gpu.mem} GB. Try a lower precision,
                    a bigger GPU, or split it across GPUs (Module 14).
                  </p>
                </div>
              </motion.div>
            )}
            <p className="mt-auto pt-4 font-mono text-[0.65rem] leading-relaxed text-ink-faint">
              Upper bound: ignores the KV cache, attention and overheads. Real systems reach 50–80% of it. Derived in{' '}
              <a href="/learn/m00/what-is-inference-engineering" className="underline">0.1</a> and{' '}
              <a href="/learn/m03/roofline" className="underline">3.3</a>.
            </p>
          </div>
        </div>
      </div>
    </MotionConfig>
  );
}
