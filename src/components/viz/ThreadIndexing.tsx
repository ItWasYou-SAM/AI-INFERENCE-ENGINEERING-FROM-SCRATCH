import { useState } from 'react';

const BLOCK_COLORS = ['var(--blue)', 'var(--red)', 'var(--green)', 'var(--amber)', 'var(--violet)', 'var(--teal)'];
const color = (b: number) => BLOCK_COLORS[b % BLOCK_COLORS.length];

function Pills<T extends number>({ label, values, value, set }: { label: string; values: T[]; value: T; set: (v: T) => void }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="eyebrow mr-1">{label}</span>
      {values.map((v) => (
        <button key={v} type="button" aria-pressed={v === value} onClick={() => set(v)}
          className={`rounded-md border px-2 py-0.5 font-mono text-xs transition-colors ${v === value ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
          {v}
        </button>
      ))}
    </div>
  );
}

function OneD() {
  const [n, setN] = useState(22);
  const [bd, setBd] = useState(8);
  const [sel, setSel] = useState(11);
  const blocks = Math.ceil(n / bd);
  const total = blocks * bd;
  const b = Math.floor(sel / bd);
  const t = sel % bd;
  const inRange = sel < n;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <Pills label="blockDim.x" values={[4, 8, 16]} value={bd} set={(v) => { setBd(v); setSel(Math.min(sel, Math.ceil(n / v) * v - 1)); }} />
        <label className="flex flex-1 items-center gap-2">
          <span className="eyebrow">n</span>
          <input type="range" min={5} max={40} value={n} onChange={(e) => { const v = +e.target.value; setN(v); setSel(Math.min(sel, Math.ceil(v / bd) * bd - 1)); }} className="flex-1" />
          <b className="w-6 font-mono text-xs">{n}</b>
        </label>
      </div>

      <p className="eyebrow mb-1.5">grid: {blocks} blocks × {bd} threads = {total} threads (click one)</p>
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: blocks }, (_, bi) => (
          <div key={bi} className="rounded-md border p-1.5" style={{ borderColor: color(bi) }}>
            <p className="mb-1 font-mono text-[0.6rem]" style={{ color: color(bi) }}>block {bi}</p>
            <div className="flex gap-0.5">
              {Array.from({ length: bd }, (_, ti) => {
                const gi = bi * bd + ti;
                const active = gi === sel;
                const idle = gi >= n;
                return (
                  <button key={ti} type="button" onClick={() => setSel(gi)} aria-label={`block ${bi} thread ${ti}`}
                    className="h-6 w-5 rounded-sm font-mono text-[0.55rem]"
                    style={{
                      background: active ? color(bi) : idle ? 'transparent' : `color-mix(in srgb, ${color(bi)} 22%, transparent)`,
                      color: active ? 'var(--paper)' : 'var(--ink)',
                      border: idle ? '1px dashed var(--rule)' : 'none',
                      opacity: idle && !active ? 0.6 : 1,
                    }}>
                    {ti}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <p className="eyebrow mt-5 mb-1.5">array x[0 … {n - 1}]</p>
      <div className="flex flex-wrap gap-0.5">
        {Array.from({ length: n }, (_, i) => (
          <div key={i} onClick={() => setSel(i)} className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-sm font-mono text-[0.6rem]"
            style={{
              background: i === sel ? color(Math.floor(i / bd)) : `color-mix(in srgb, ${color(Math.floor(i / bd))} 12%, transparent)`,
              color: i === sel ? 'var(--paper)' : 'var(--ink-soft)',
            }}>
            {i}
          </div>
        ))}
      </div>

      <div className="mt-4 rounded-md border border-rule bg-paper p-3 font-mono text-sm">
        i = blockIdx.x × blockDim.x + threadIdx.x = <span style={{ color: color(b) }}>{b}</span> × {bd} + {t} = <b>{sel}</b>
        <p className="mt-1 font-serif text-sm text-ink-soft">
          {inRange ? (
            <>This thread handles <code>x[{sel}]</code>.</>
          ) : (
            <>
              <b style={{ color: 'var(--red)' }}>Out of range.</b> The last block is rounded up, so {total - n} thread{total - n === 1 ? '' : 's'} have no element. The{' '}
              <code>if (i &lt; n)</code> guard stops them from writing past the end of the array.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function TwoD() {
  const ROWS = 6;
  const COLS = 10;
  const [bx, setBx] = useState(4);
  const [by, setBy] = useState(2);
  const [sel, setSel] = useState<[number, number]>([3, 6]);
  const gx = Math.ceil(COLS / bx);
  const gy = Math.ceil(ROWS / by);
  const [r, c] = sel;
  const bIdx = [Math.floor(c / bx), Math.floor(r / by)];
  const tIdx = [c % bx, r % by];
  const blockNum = bIdx[1] * gx + bIdx[0];

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-4">
        <Pills label="blockDim.x" values={[2, 4, 8]} value={bx} set={setBx} />
        <Pills label="blockDim.y" values={[1, 2, 3]} value={by} set={setBy} />
        <span className="ml-auto font-mono text-[0.7rem] text-ink-faint">grid = ({gx}, {gy}) blocks</span>
      </div>
      <p className="eyebrow mb-1.5">a {ROWS} × {COLS} row-major matrix; colour = block (click a cell)</p>
      <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${COLS}, 2rem)` }}>
        {Array.from({ length: ROWS * COLS }, (_, k) => {
          const rr = Math.floor(k / COLS);
          const cc = k % COLS;
          const bn = Math.floor(rr / by) * gx + Math.floor(cc / bx);
          const active = rr === r && cc === c;
          return (
            <button key={k} type="button" onClick={() => setSel([rr, cc])} aria-label={`row ${rr} col ${cc}`}
              className="h-8 rounded-sm font-mono text-[0.6rem]"
              style={{
                background: active ? color(bn) : `color-mix(in srgb, ${color(bn)} ${bn % 2 ? 26 : 14}%, transparent)`,
                color: active ? 'var(--paper)' : 'var(--ink-soft)',
              }}>
              {k}
            </button>
          );
        })}
      </div>
      <div className="mt-4 rounded-md border border-rule bg-paper p-3 font-mono text-xs leading-relaxed">
        <p>blockIdx = ({bIdx[0]}, {bIdx[1]})   threadIdx = ({tIdx[0]}, {tIdx[1]})</p>
        <p>col = blockIdx.x × blockDim.x + threadIdx.x = {bIdx[0]} × {bx} + {tIdx[0]} = <b>{c}</b></p>
        <p>row = blockIdx.y × blockDim.y + threadIdx.y = {bIdx[1]} × {by} + {tIdx[1]} = <b>{r}</b></p>
        <p>index = row × cols + col = {r} × {COLS} + {c} = <b style={{ color: color(blockNum) }}>{r * COLS + c}</b></p>
        <p className="mt-1 font-serif text-sm text-ink-soft">
          x runs along columns, so neighbouring threads in a warp touch neighbouring addresses (<a href="/learn/m04/coalesced-vs-strided">4.7</a>).
        </p>
      </div>
    </div>
  );
}

export default function ThreadIndexing() {
  const [mode, setMode] = useState<'1d' | '2d'>('1d');
  return (
    <div className="not-prose text-ink">
      <div className="mb-4 flex gap-2">
        {(['1d', '2d'] as const).map((m) => (
          <button key={m} type="button" aria-pressed={m === mode} onClick={() => setMode(m)}
            className={`rounded-md border px-3 py-1 font-mono text-xs transition-colors ${m === mode ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'}`}>
            {m === '1d' ? '1D: a vector' : '2D: a matrix'}
          </button>
        ))}
      </div>
      {mode === '1d' ? <OneD /> : <TwoD />}
    </div>
  );
}
