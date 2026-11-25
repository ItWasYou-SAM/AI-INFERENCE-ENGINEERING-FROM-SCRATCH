import { useState } from 'react';
import { MotionConfig } from 'motion/react';

/** Public configs (config.json on Hugging Face). ff = intermediate_size of the SwiGLU MLP. */
const MODELS = [
  { name: 'Qwen2.5-0.5B', d: 896, heads: 14, kv: 2, ff: 4864, layers: 24, vocab: 151936, tied: true },
  { name: 'Llama-3-8B', d: 4096, heads: 32, kv: 8, ff: 14336, layers: 32, vocab: 128256, tied: false },
  { name: 'Llama-3-70B', d: 8192, heads: 64, kv: 8, ff: 28672, layers: 80, vocab: 128256, tied: false },
];
type Model = (typeof MODELS)[number];

type Kind = 'norm' | 'matmul' | 'attn' | 'add' | 'act';
interface Block {
  id: string;
  label: string;
  kind: Kind;
  group: 'attn' | 'mlp' | 'io';
  shape: (m: Model) => string;
  params: (m: Model) => number;
  /** FLOPs per token, excluding attention-over-context (which depends on context length). */
  flops: (m: Model) => number;
  what: string;
}

const hd = (m: Model) => m.d / m.heads;
const BLOCKS: Block[] = [
  { id: 'in', label: 'x  (residual stream)', kind: 'add', group: 'io', shape: (m) => `[tokens, ${m.d}]`, params: () => 0, flops: () => 0,
    what: 'One vector of d_model numbers per token. Every layer reads it, adds something, and passes it on.' },
  { id: 'n1', label: 'RMSNorm', kind: 'norm', group: 'attn', shape: (m) => `[tokens, ${m.d}]`, params: (m) => m.d, flops: (m) => 4 * m.d,
    what: 'Rescale each token vector to unit RMS. Elementwise and cheap in FLOPs, but it still reads and writes the whole activation.' },
  { id: 'qkv', label: 'Q, K, V projections', kind: 'matmul', group: 'attn',
    shape: (m) => `${m.d} → ${m.d} + 2×${m.kv * hd(m)}`,
    params: (m) => m.d * m.d + 2 * m.d * m.kv * hd(m), flops: (m) => 2 * (m.d * m.d + 2 * m.d * m.kv * hd(m)),
    what: 'Three matmuls (often fused into one). K and V are what the KV cache stores. With GQA there are fewer K/V heads than Q heads, so K and V are narrower.' },
  { id: 'att', label: 'Attention  softmax(QKᵀ)V', kind: 'attn', group: 'attn', shape: (m) => `${m.heads} heads × ${hd(m)} dims`, params: () => 0,
    flops: () => 0,
    what: 'The only step where tokens look at each other. No weights, but its cost grows with context length: each new token reads K and V for every earlier position.' },
  { id: 'o', label: 'Output projection', kind: 'matmul', group: 'attn', shape: (m) => `${m.d} → ${m.d}`, params: (m) => m.d * m.d, flops: (m) => 2 * m.d * m.d,
    what: 'Mixes the heads back into one d_model vector.' },
  { id: 'r1', label: '+ residual', kind: 'add', group: 'attn', shape: (m) => `[tokens, ${m.d}]`, params: () => 0, flops: (m) => m.d,
    what: 'Add the attention output back onto the stream. The layer only ever adds; it never overwrites.' },
  { id: 'n2', label: 'RMSNorm', kind: 'norm', group: 'mlp', shape: (m) => `[tokens, ${m.d}]`, params: (m) => m.d, flops: (m) => 4 * m.d,
    what: 'Same as before, with its own learned scale.' },
  { id: 'up', label: 'gate + up projections', kind: 'matmul', group: 'mlp', shape: (m) => `${m.d} → 2×${m.ff}`, params: (m) => 2 * m.d * m.ff,
    flops: (m) => 4 * m.d * m.ff, what: 'Two wide matmuls. The MLP is where most of the parameters live.' },
  { id: 'act', label: 'SiLU(gate) × up', kind: 'act', group: 'mlp', shape: (m) => `[tokens, ${m.ff}]`, params: () => 0, flops: (m) => 5 * m.ff,
    what: 'An elementwise nonlinearity. Without it, stacking layers would collapse into one big linear map.' },
  { id: 'down', label: 'down projection', kind: 'matmul', group: 'mlp', shape: (m) => `${m.ff} → ${m.d}`, params: (m) => m.ff * m.d, flops: (m) => 2 * m.ff * m.d,
    what: 'Back down to d_model.' },
  { id: 'r2', label: '+ residual', kind: 'add', group: 'mlp', shape: (m) => `[tokens, ${m.d}]`, params: () => 0, flops: (m) => m.d,
    what: 'Add the MLP output onto the stream. That is the whole layer; the next one starts here.' },
];

const KIND_COLOR: Record<Kind, string> = {
  matmul: 'var(--blue)',
  attn: 'var(--violet)',
  norm: 'var(--teal)',
  act: 'var(--amber)',
  add: 'var(--ink-faint)',
};

const fmt = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} K` : `${n}`;

export default function TransformerLayer() {
  const [model, setModel] = useState(MODELS[1]);
  const [sel, setSel] = useState('qkv');
  const block = BLOCKS.find((b) => b.id === sel)!;

  const attnP = BLOCKS.filter((b) => b.group === 'attn').reduce((s, b) => s + b.params(model), 0);
  const mlpP = BLOCKS.filter((b) => b.group === 'mlp').reduce((s, b) => s + b.params(model), 0);
  const layerP = attnP + mlpP;
  const embedP = model.vocab * model.d * (model.tied ? 1 : 2);
  const totalP = layerP * model.layers + embedP + model.d;
  const matmulShare = BLOCKS.filter((b) => b.kind === 'matmul').reduce((s, b) => s + b.flops(model), 0) /
    BLOCKS.reduce((s, b) => s + b.flops(model), 0);

  return (
    <MotionConfig reducedMotion="user">
      <div className="not-prose text-ink">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="eyebrow mr-1">Model</span>
          {MODELS.map((m) => (
            <button
              key={m.name}
              type="button"
              aria-pressed={m === model}
              onClick={() => setModel(m)}
              className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${
                m === model ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'
              }`}
            >
              {m.name}
            </button>
          ))}
          <span className="ml-auto font-mono text-[0.7rem] text-ink-faint">
            ×{model.layers} layers · ≈{fmt(totalP)} params
          </span>
        </div>

        <div className="grid gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          {/* the layer, bottom to top as in most diagrams: we list top to bottom in data-flow order */}
          <ol className="relative space-y-1.5" aria-label="Blocks of one transformer layer, in data-flow order">
            {BLOCKS.map((b, i) => {
              const active = b.id === sel;
              const share = b.params(model) / layerP;
              return (
                <li key={b.id}>
                  {(i === 1 || i === 6) && (
                    <p className="eyebrow mt-2 mb-1 text-ink-faint">{i === 1 ? 'attention sub-block' : 'MLP sub-block'}</p>
                  )}
                  <button
                    type="button"
                    onClick={() => setSel(b.id)}
                    aria-pressed={active}
                    className={`relative flex w-full items-center gap-3 overflow-hidden rounded-md border px-3 py-1.5 text-left text-sm transition-colors ${
                      active ? 'border-ink bg-paper' : 'border-rule bg-paper/60 hover:border-ink-faint'
                    }`}
                  >
                    <span
                      aria-hidden="true"
                      className="absolute inset-y-0 left-0"
                      style={{ width: `${share * 100}%`, transition: 'width 0.4s ease',  background: KIND_COLOR[b.kind], opacity: 0.13 }}
                    />
                    <span className="relative h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: KIND_COLOR[b.kind] }} />
                    <span className="relative flex-1">{b.label}</span>
                    <span className="relative font-mono text-[0.7rem] text-ink-faint">
                      {b.params(model) ? `${(share * 100).toFixed(1)}%` : ''}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>

          <div className="flex flex-col gap-4">
            <div className="rounded-md border border-rule bg-paper p-4">
              <p className="eyebrow" style={{ color: KIND_COLOR[block.kind] }}>
                {block.kind === 'matmul' ? 'matmul: reads weights' : block.kind === 'attn' ? 'attention: reads KV' : 'elementwise'}
              </p>
              <h4 className="mt-1 font-serif text-lg">{block.label}</h4>
              <p className="mt-2 text-sm text-ink-soft">{block.what}</p>
              <dl className="mt-3 grid grid-cols-2 gap-2 font-mono text-xs">
                <div className="col-span-2">
                  <dt className="text-ink-faint">shape</dt>
                  <dd className="text-ink">{block.shape(model)}</dd>
                </div>
                <div>
                  <dt className="text-ink-faint">params</dt>
                  <dd className="text-ink">{fmt(block.params(model))}</dd>
                </div>
                <div>
                  <dt className="text-ink-faint">FLOPs/token</dt>
                  <dd className="text-ink">{block.kind === 'attn' ? '∝ context' : fmt(block.flops(model))}</dd>
                </div>
              </dl>
            </div>

            <div className="rounded-md border border-rule bg-paper p-4">
              <p className="eyebrow">Where one layer's weights live</p>
              <div className="mt-3 flex h-6 overflow-hidden rounded" role="img" aria-label={`Attention ${((attnP / layerP) * 100).toFixed(0)} percent, MLP ${((mlpP / layerP) * 100).toFixed(0)} percent`}>
                <div style={{ width: `${(attnP / layerP) * 100}%`, transition: 'width 0.4s ease',  background: 'var(--violet)' }} />
                <div style={{ width: `${(mlpP / layerP) * 100}%`, transition: 'width 0.4s ease',  background: 'var(--blue)' }} />
              </div>
              <div className="mt-2 flex justify-between font-mono text-xs">
                <span style={{ color: 'var(--violet)' }}>attention {((attnP / layerP) * 100).toFixed(0)}% · {fmt(attnP)}</span>
                <span style={{ color: 'var(--blue)' }}>MLP {((mlpP / layerP) * 100).toFixed(0)}% · {fmt(mlpP)}</span>
              </div>
              <p className="mt-3 text-sm text-ink-soft">
                Matmuls are <b className="text-ink">{(matmulShare * 100).toFixed(2)}%</b> of the per-token FLOPs outside attention. Every one
                of them reads its whole weight matrix from memory, once per step.
              </p>
            </div>
          </div>
        </div>
        <p className="mt-3 font-mono text-[0.7rem] text-ink-faint">
          Bar width = share of the layer's parameters. Embeddings ({fmt(embedP)}{model.tied ? ', tied' : ', untied'}) sit outside the layers.
        </p>
      </div>
    </MotionConfig>
  );
}
