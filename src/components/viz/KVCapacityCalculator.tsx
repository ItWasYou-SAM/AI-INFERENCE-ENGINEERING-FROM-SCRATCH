import { useMemo, useState } from 'react';
import { motion, MotionConfig } from 'motion/react';

/* Decimal units (1 GB = 1e9 B), matching the MOHITH source's arithmetic. */
const GB = 1e9;

interface ModelSpec {
  name: string;
  params: number; // billions
  layers: number;
  kvHeads: number;
  qHeads: number;
  headDim: number;
}

const MODELS: ModelSpec[] = [
  { name: 'Mistral 7B', params: 7.24, layers: 32, kvHeads: 8, qHeads: 32, headDim: 128 },
  { name: 'Llama 2 7B (MHA)', params: 6.74, layers: 32, kvHeads: 32, qHeads: 32, headDim: 128 },
  { name: 'Llama 3.1 8B', params: 8.03, layers: 32, kvHeads: 8, qHeads: 32, headDim: 128 },
  { name: 'Llama 3.1 70B', params: 70.6, layers: 80, kvHeads: 8, qHeads: 64, headDim: 128 },
];

const GPUS = [
  { name: 'RTX 4090', mem: 24 },
  { name: 'L4', mem: 24 },
  { name: 'A100-80GB', mem: 80 },
  { name: 'H100 80GB', mem: 80 },
  { name: 'H200', mem: 141 },
];

const WEIGHT_FMT = [
  { name: 'FP16', bytes: 2 },
  { name: 'FP8', bytes: 1 },
  { name: 'INT4', bytes: 0.5 },
];
const KV_FMT = [
  { name: 'FP16', bytes: 2 },
  { name: 'FP8', bytes: 1 },
];

const CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

const fmtGB = (b: number) => (b / GB >= 100 ? (b / GB).toFixed(0) : (b / GB).toFixed(b / GB >= 10 ? 1 : 2)) + ' GB';
const fmtCtx = (n: number) => (n >= 1024 ? `${n / 1024}K` : `${n}`);

function Segmented<T>({
  label,
  options,
  value,
  onChange,
  render,
}: {
  label: string;
  options: T[];
  value: T;
  onChange: (v: T) => void;
  render: (v: T) => string;
}) {
  return (
    <fieldset className="min-w-0">
      <legend className="eyebrow mb-1.5">{label}</legend>
      <div className="flex flex-wrap gap-1">
        {options.map((o) => {
          const on = o === value;
          return (
            <button
              key={render(o)}
              type="button"
              aria-pressed={on}
              onClick={() => onChange(o)}
              className={`relative rounded-md px-2.5 py-1 font-mono text-xs transition-colors ${
                on ? 'text-paper' : 'text-ink-soft hover:text-ink'
              }`}
            >
              {on && (
                <motion.span
                  layoutId={`seg-${label}`}
                  className="absolute inset-0 rounded-md bg-ink"
                  transition={{ type: 'spring', stiffness: 500, damping: 38 }}
                />
              )}
              <span className="relative">{render(o)}</span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

export default function KVCapacityCalculator() {
  const [model, setModel] = useState(MODELS[0]);
  const [gpu, setGpu] = useState(GPUS[2]);
  const [wf, setWf] = useState(WEIGHT_FMT[0]);
  const [kf, setKf] = useState(KV_FMT[0]);
  const [ctxIdx, setCtxIdx] = useState(3); // 4K
  const [users, setUsers] = useState(80);
  const [overheadGB, setOverheadGB] = useState(8);

  const ctx = CONTEXTS[ctxIdx];

  const m = useMemo(() => {
    const kvPerToken = 2 * model.layers * model.kvHeads * model.headDim * kf.bytes;
    const kvPerUser = kvPerToken * ctx;
    const weights = model.params * 1e9 * wf.bytes;
    const overhead = overheadGB * GB;
    const cap = gpu.mem * GB;
    const free = cap - weights - overhead;
    const maxUsers = Math.max(0, Math.floor(free / kvPerUser));
    const kv = kvPerUser * users;
    const total = weights + overhead + kv;
    return { kvPerToken, kvPerUser, weights, overhead, cap, free, maxUsers, kv, total, oom: total > cap };
  }, [model, gpu, wf, kf, ctx, users, overheadGB]);

  const scale = Math.max(m.cap, m.total);
  const pct = (b: number) => `${(Math.max(0, b) / scale) * 100}%`;

  return (
    <MotionConfig reducedMotion="user">
    <div className="not-prose text-ink">
      <div className="grid gap-4 sm:grid-cols-2">
        <Segmented label="Model" options={MODELS} value={model} onChange={setModel} render={(v) => v.name} />
        <Segmented label="GPU" options={GPUS} value={gpu} onChange={setGpu} render={(v) => `${v.name}`} />
        <Segmented label="Weights" options={WEIGHT_FMT} value={wf} onChange={setWf} render={(v) => v.name} />
        <Segmented label="KV cache" options={KV_FMT} value={kf} onChange={setKf} render={(v) => v.name} />
      </div>

      <div className="mt-5 grid gap-5 sm:grid-cols-3">
        <label className="block">
          <span className="eyebrow flex justify-between">
            Context length <b className="text-ink">{fmtCtx(ctx)} tokens</b>
          </span>
          <input
            type="range"
            min={0}
            max={CONTEXTS.length - 1}
            value={ctxIdx}
            onChange={(e) => setCtxIdx(+e.target.value)}
            className="mt-2 w-full accent-[var(--blue)]"
          />
        </label>
        <label className="block">
          <span className="eyebrow flex justify-between">
            Concurrent users <b className="text-ink">{users}</b>
          </span>
          <input
            type="range"
            min={1}
            max={256}
            value={users}
            onChange={(e) => setUsers(+e.target.value)}
            className="mt-2 w-full accent-[var(--blue)]"
          />
        </label>
        <label className="block">
          <span className="eyebrow flex justify-between">
            Overhead <b className="text-ink">{overheadGB} GB</b>
          </span>
          <input
            type="range"
            min={0}
            max={20}
            value={overheadGB}
            onChange={(e) => setOverheadGB(+e.target.value)}
            className="mt-2 w-full accent-[var(--blue)]"
          />
        </label>
      </div>

      {/* Per-token formula with live numbers */}
      <div className="mt-6 overflow-x-auto rounded-md border border-dashed border-rule bg-paper px-4 py-3 font-mono text-[0.8rem] whitespace-nowrap">
        <span className="text-ink-faint">KV / token = </span>2 <span className="text-ink-faint">(K,V)</span> × {model.layers}{' '}
        <span className="text-ink-faint">layers</span> × {model.kvHeads} <span className="text-ink-faint">kv-heads</span> ×{' '}
        {model.headDim} <span className="text-ink-faint">dim</span> × {kf.bytes} <span className="text-ink-faint">B</span> ={' '}
        <b className="text-blue">{(m.kvPerToken / 1e3).toFixed(0)} KB</b>
        <span className="text-ink-faint"> · per user @ {fmtCtx(ctx)} = </span>
        <b className="text-blue">{fmtGB(m.kvPerUser)}</b>
      </div>

      {/* Memory bar */}
      <div className="mt-7">
        <div className="mb-2 flex items-baseline justify-between">
          <span className="eyebrow">GPU memory: {gpu.name}</span>
          <span className="font-mono text-xs text-ink-soft">
            used {fmtGB(m.total)} / {gpu.mem} GB
          </span>
        </div>
        <motion.div
          className="relative h-16 overflow-hidden rounded-md border border-ink/70 bg-paper"
          animate={m.oom ? { x: [0, -5, 5, -3, 3, 0] } : { x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="absolute inset-y-0 left-0 flex w-full">
            <div
              className="flex h-full items-center justify-center overflow-hidden bg-ink font-mono text-[0.65rem] text-paper transition-[width] duration-500 ease-out"
              style={{ width: pct(m.weights) }}
              title="Model weights"
            >
              weights
            </div>
            <div
              className="h-full transition-[width] duration-500 ease-out"
              style={{
                width: pct(m.overhead),
                background: 'repeating-linear-gradient(45deg, var(--ink-faint) 0 2px, transparent 2px 7px)',
                opacity: 0.6,
              }}
              title="Activations, CUDA context, fragmentation"
            />
            <div
              className="h-full transition-[width] duration-500 ease-out"
              style={{
                width: pct(m.kv),
                backgroundColor: m.oom ? 'color-mix(in srgb, var(--red) 55%, transparent)' : 'color-mix(in srgb, var(--blue) 55%, transparent)',
                // one stripe per user, so the bar reads as "N copies of one user's KV"
                backgroundImage: users <= 96 ? 'linear-gradient(90deg, transparent calc(100% - 1.5px), var(--paper) calc(100% - 1.5px))' : 'none',
                backgroundSize: `${100 / users}% 100%`,
              }}
              title={`KV cache: ${users} users`}
            />
          </div>
          {/* capacity line */}
          <div className="absolute inset-y-0 border-r-2 border-dashed border-red transition-[left] duration-500" style={{ left: `calc(${pct(m.cap)} - 2px)` }} />
        </motion.div>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[0.7rem] text-ink-soft">
          <span>
            <i className="mr-1 inline-block h-2 w-3 bg-ink align-middle" /> weights {fmtGB(m.weights)}
          </span>
          <span>
            <i className="mr-1 inline-block h-2 w-3 bg-ink-faint/60 align-middle" /> overhead {fmtGB(m.overhead)}
          </span>
          <span>
            <i className="mr-1 inline-block h-2 w-3 align-middle" style={{ background: 'color-mix(in srgb, var(--blue) 55%, transparent)' }} /> KV
            cache {fmtGB(m.kv)} ({users} × {fmtGB(m.kvPerUser)})
          </span>
          <span className="text-red">┆ capacity {gpu.mem} GB</span>
        </div>
      </div>

      {/* Verdict */}
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <Stat label="Max users at this context" value={m.free <= 0 ? '0' : String(m.maxUsers)} tone={m.maxUsers === 0 ? 'red' : 'blue'} />
        <Stat label="Memory left for KV" value={m.free <= 0 ? 'none' : fmtGB(m.free)} />
        <Stat
          label="Status"
          value={m.free <= 0 ? 'Weights don’t fit' : m.oom ? `OOM by ${fmtGB(m.total - m.cap)}` : 'Fits'}
          tone={m.oom || m.free <= 0 ? 'red' : 'green'}
        />
      </div>
    </div>
    </MotionConfig>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'red' | 'green' | 'blue' }) {
  const color = tone ? `var(--${tone})` : 'var(--ink)';
  return (
    <div className="rounded-md border border-rule bg-paper px-4 py-3">
      <div className="eyebrow">{label}</div>
      <motion.div
        key={value}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-1 font-display text-2xl tabular-nums"
        style={{ color }}
      >
        {value}
      </motion.div>
    </div>
  );
}
