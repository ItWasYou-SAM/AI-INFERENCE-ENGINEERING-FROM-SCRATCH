import { useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'motion/react';

interface Layer {
  name: string;
  question: string;
  examples: string[];
  knobs: string[];
  modules: { id: string; label: string; href: string }[];
  color: string;
}

/* Top of the stack = closest to the user; bottom = closest to silicon. */
const LAYERS: Layer[] = [
  {
    name: 'Fleet & economics',
    question: 'What does a token cost, and are we meeting our SLOs?',
    examples: ['cost per million tokens', 'SLO goodput', 'capacity planning', 'llm-d control plane'],
    knobs: ['GPU type per workload', 'autoscaling policy', 'cost-aware routing'],
    modules: [{ id: 'M16', label: 'Economics & control plane', href: '/learn/m16/cost-per-million-tokens' }],
    color: 'var(--amber)',
  },
  {
    name: 'Serving & routing',
    question: 'Which replica gets this request, and where does its KV cache live?',
    examples: ['vLLM Router', 'prefill/decode disaggregation', 'Kubernetes', 'KV-affinity routing'],
    knobs: ['load-balancing policy', 'P:D ratio', 'replicas vs tensor parallelism'],
    modules: [
      { id: 'M14', label: 'Multi-GPU parallelism', href: '/learn/m14/collectives-nccl' },
      { id: 'M15', label: 'Distributed serving', href: '/learn/m15/why-disaggregate' },
    ],
    color: 'var(--red)',
  },
  {
    name: 'Inference engine',
    question: 'Which requests run on this step, and where is each one’s KV cache?',
    examples: ['vLLM', 'SGLang', 'TensorRT-LLM', 'continuous batching', 'PagedAttention'],
    knobs: ['max batch size', 'chunked prefill', 'prefix caching', 'speculative decoding'],
    modules: [
      { id: 'M9', label: 'KV cache', href: '/learn/m09/kv-cache-size' },
      { id: 'M10', label: 'Engine internals', href: '/learn/m10/continuous-batching' },
      { id: 'M11', label: 'Decoding', href: '/learn/m11/sampling-strategies' },
      { id: 'M13', label: 'Benchmarking', href: '/learn/m13/serving-metrics' },
    ],
    color: 'var(--blue)',
  },
  {
    name: 'Model optimizations',
    question: 'Can the same model do less work, or move fewer bytes?',
    examples: ['INT4 / FP8 quantization', 'grouped-query attention', 'FlashAttention', 'torch.compile'],
    knobs: ['weight & KV precision', 'attention variant', 'fusion'],
    modules: [
      { id: 'M8', label: 'Transformer kernels', href: '/learn/m08/norms-fused-residual' },
      { id: 'M12', label: 'Quantization', href: '/learn/m12/number-formats' },
    ],
    color: 'var(--green)',
  },
  {
    name: 'Kernels',
    question: 'How close does each operation get to the hardware’s limits?',
    examples: ['CUDA', 'Triton', 'GEMM tiling', 'reductions', 'Nsight profiles'],
    knobs: ['tile sizes', 'memory coalescing', 'occupancy', 'tensor cores'],
    modules: [
      { id: 'M4', label: 'CUDA fundamentals', href: '/learn/m04/kernels-are-not-functions' },
      { id: 'M5', label: 'Parallel patterns', href: '/learn/m05/reduction-trees' },
      { id: 'M6', label: 'GEMM', href: '/learn/m06/tiled-matmul' },
      { id: 'M7', label: 'Profiling', href: '/learn/m07/top-down-profiling' },
    ],
    color: 'var(--ink-soft)',
  },
  {
    name: 'Hardware',
    question: 'How many FLOPs and how many bytes per second do we actually have?',
    examples: ['SMs & warps', 'HBM bandwidth', 'tensor cores', 'NVLink'],
    knobs: ['which GPU', 'how many', 'how they are connected'],
    modules: [
      { id: 'M2', label: 'GPU first principles', href: '/learn/m02/why-not-10000-cpu-cores' },
      { id: 'M3', label: 'Roofline', href: '/learn/m03/roofline' },
    ],
    color: 'var(--ink)',
  },
];

export default function InferenceStack() {
  const [active, setActive] = useState(2);
  const L = LAYERS[active];

  return (
    <MotionConfig reducedMotion="user">
      <div className="not-prose grid gap-6 text-ink md:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* the stack */}
        <div className="relative pl-6">
          <div className="absolute top-0 bottom-0 left-0 flex flex-col justify-between py-1 font-mono text-[0.6rem] tracking-widest text-ink-faint uppercase [writing-mode:vertical-rl] rotate-180">
            <span>silicon</span>
            <span>user</span>
          </div>
          <ol className="flex flex-col gap-1.5" role="tablist" aria-label="Layers of the inference stack">
            {LAYERS.map((layer, i) => {
              const on = i === active;
              return (
                <motion.li
                  key={layer.name}
                  initial={{ opacity: 0, x: -16 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: (LAYERS.length - i) * 0.07, duration: 0.45 }}
                >
                  <button
                    type="button"
                    role="tab"
                    aria-selected={on}
                    onClick={() => setActive(i)}
                    onMouseEnter={() => setActive(i)}
                    className="relative block w-full rounded-md border border-dashed px-4 py-2.5 text-left transition-colors"
                    style={{
                      borderColor: on ? layer.color : 'var(--rule)',
                      background: on ? `color-mix(in srgb, ${layer.color} 10%, var(--paper))` : 'var(--paper)',
                    }}
                  >
                    {on && (
                      <motion.span
                        layoutId="stack-marker"
                        className="absolute top-2 bottom-2 -left-[1.1rem] w-1 rounded-full"
                        style={{ background: layer.color }}
                        transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                      />
                    )}
                    <span className="flex items-baseline justify-between gap-3">
                      <span className="font-display text-[1.05rem]">{layer.name}</span>
                      <span className="font-mono text-[0.62rem] text-ink-faint">{layer.modules.map((m) => m.id).join(' · ')}</span>
                    </span>
                  </button>
                </motion.li>
              );
            })}
          </ol>
        </div>

        {/* the detail card */}
        <div className="relative min-h-[17rem]">
          <AnimatePresence mode="wait">
            <motion.div
              key={L.name}
              initial={{ opacity: 0, y: 10, rotate: -0.4 }}
              animate={{ opacity: 1, y: 0, rotate: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              className="sheet h-full p-5"
              role="tabpanel"
            >
              <p className="eyebrow" style={{ color: L.color }}>
                {L.name}
              </p>
              <p className="mt-2 font-display text-xl leading-snug">“{L.question}”</p>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <div>
                  <p className="eyebrow mb-1.5">Lives here</p>
                  <ul className="space-y-1 text-sm">
                    {L.examples.map((e) => (
                      <li key={e} className="flex gap-2">
                        <span className="text-ink-faint">–</span>
                        {e}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="eyebrow mb-1.5">Knobs you turn</p>
                  <ul className="space-y-1 text-sm">
                    {L.knobs.map((k) => (
                      <li key={k} className="flex gap-2">
                        <span style={{ color: L.color }}>◆</span>
                        {k}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="mt-5 flex flex-wrap gap-2">
                {L.modules.map((m) => (
                  <a
                    key={m.id}
                    href={m.href}
                    className="rounded-full border border-rule px-2.5 py-1 font-mono text-[0.68rem] text-ink-soft transition-colors hover:border-ink hover:text-ink"
                  >
                    {m.id} · {m.label} →
                  </a>
                ))}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </MotionConfig>
  );
}
