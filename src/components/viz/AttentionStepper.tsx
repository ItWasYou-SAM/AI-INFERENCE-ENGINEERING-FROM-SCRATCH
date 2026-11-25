import { useMemo, useState } from 'react';
import { MotionConfig } from 'motion/react';

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];

/** Hand-picked scaled scores q·k/√d, chosen so the weights look like a plausible single head. Row = query, column = key. */
const SCORES = [
  [1.0, 0.2, -0.3, 0.1, 0.4, -0.5],
  [1.2, 1.5, 0.6, -0.2, 0.3, 0.9],
  [0.3, 2.2, 1.1, 0.4, -0.1, 0.8],
  [-0.4, 0.6, 1.9, 0.9, 0.2, 1.4],
  [1.8, 0.1, -0.2, 0.7, 1.0, 0.6],
  [0.2, 1.6, 0.9, 1.3, 0.8, 1.1],
];

function softmaxRow(row: number[], visible: boolean[]) {
  const vals = row.map((v, j) => (visible[j] ? v : -Infinity));
  const mx = Math.max(...vals);
  const e = vals.map((v) => (v === -Infinity ? 0 : Math.exp(v - mx)));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

const STAGES = ['scores', 'mask', 'softmax', 'weighted sum'] as const;

export default function AttentionStepper() {
  const [q, setQ] = useState(3);
  const [stage, setStage] = useState(2);
  const [causal, setCausal] = useState(true);
  const N = TOKENS.length;

  const weights = useMemo(
    () => SCORES.map((row, i) => softmaxRow(row, row.map((_, j) => !causal || j <= i))),
    [causal],
  );

  const cell = (i: number, j: number) => {
    const hidden = causal && j > i && stage >= 1;
    if (hidden) return { text: '−∞', bg: 'transparent', fg: 'var(--ink-faint)' };
    if (stage === 0 || stage === 1) {
      const v = SCORES[i][j];
      const a = Math.min(1, Math.abs(v) / 2.2);
      return { text: v.toFixed(1), bg: `color-mix(in srgb, ${v >= 0 ? 'var(--blue)' : 'var(--red)'} ${a * 45}%, transparent)`, fg: 'var(--ink)' };
    }
    const w = weights[i][j];
    return { text: w.toFixed(2), bg: `color-mix(in srgb, var(--violet) ${Math.round(w * 85)}%, transparent)`, fg: w > 0.45 ? 'var(--paper)' : 'var(--ink)' };
  };

  const row = weights[q];
  const leaks = !causal && row.slice(q + 1).reduce((a, b) => a + b, 0);

  return (
    <MotionConfig reducedMotion="user">
      <div className="not-prose text-ink">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="eyebrow mr-1">Stage</span>
          {STAGES.map((s, i) => (
            <button
              key={s}
              type="button"
              aria-pressed={i === stage}
              onClick={() => setStage(i)}
              className={`rounded-md border px-2.5 py-1 font-mono text-xs transition-colors ${
                i === stage ? 'border-ink bg-ink text-paper' : 'border-rule text-ink-soft hover:text-ink'
              }`}
            >
              {i + 1}. {s}
            </button>
          ))}
          <label className="ml-auto flex items-center gap-2 font-mono text-xs text-ink-soft">
            <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} style={{ accentColor: 'var(--red)' }} />
            causal mask
          </label>
        </div>

        <div className="flex flex-col gap-5">
          {/* the matrix */}
          <div className="overflow-x-auto">
            <table className="border-separate border-spacing-1 font-mono text-xs" aria-label="Attention matrix: rows are queries, columns are keys">
              <thead>
                <tr>
                  <th className="text-right text-[0.65rem] font-normal text-ink-faint">query ↓ key →</th>
                  {TOKENS.map((t, j) => (
                    <th
                      key={j}
                      className="px-1 pb-1 font-normal"
                      style={{ color: j <= q || !causal ? 'var(--ink)' : 'var(--ink-faint)' }}
                    >
                      {t}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SCORES.map((_, i) => (
                  <tr key={i}>
                    <th className="pr-2 text-right font-normal">
                      <button
                        type="button"
                        onClick={() => setQ(i)}
                        className={`rounded px-1.5 py-0.5 ${i === q ? 'bg-ink text-paper' : 'text-ink-soft hover:text-ink'}`}
                      >
                        {TOKENS[i]}
                      </button>
                    </th>
                    {TOKENS.map((_, j) => {
                      const c = cell(i, j);
                      return (
                        <td
                          key={j}
                          onClick={() => setQ(i)}
                          className="h-9 w-12 cursor-pointer rounded text-center transition-colors"
                          style={{
                            background: c.bg,
                            color: c.fg,
                            outline: i === q ? '1.5px solid var(--ink)' : '1px dashed var(--rule)',
                            opacity: i === q ? 1 : 0.55,
                          }}
                        >
                          {c.text}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-2 flex items-center gap-2">
              <button type="button" onClick={() => setQ(Math.max(0, q - 1))} className="rounded-md border border-rule px-2 py-0.5 font-mono text-xs hover:border-ink">
                ← prev
              </button>
              <button type="button" onClick={() => setQ(Math.min(N - 1, q + 1))} className="rounded-md border border-rule px-2 py-0.5 font-mono text-xs hover:border-ink">
                next →
              </button>
              <span className="font-mono text-[0.7rem] text-ink-faint">query = “{TOKENS[q]}” (position {q})</span>
            </div>
          </div>

          {/* explanation for the selected query */}
          <div className="rounded-md border border-rule bg-paper p-4">
            <p className="eyebrow">
              Stage {stage + 1}: {STAGES[stage]}
            </p>
            <p className="mt-2 text-sm text-ink-soft">
              {stage === 0 && <>Each query vector is dotted with every key vector and divided by √d_head. A big score means “this key is relevant to me”.</>}
              {stage === 1 && (causal ? <>Scores for future positions are set to −∞. A token may only look at itself and what came before it, because during generation the future does not exist yet.</> : <>The mask is off, so “{TOKENS[q]}” can see tokens that come after it. Training on that would teach the model to cheat.</>)}
              {stage === 2 && <>Softmax turns each row into weights that sum to 1. Masked entries become exactly 0.</>}
              {stage === 3 && <>The output for “{TOKENS[q]}” is the weighted sum of the value vectors, using these weights.</>}
            </p>

            <div className="mt-4 space-y-1.5" aria-label={`Attention weights for query ${TOKENS[q]}`}>
              {TOKENS.map((t, j) => {
                const masked = causal && j > q;
                return (
                  <div key={j} className="flex items-center gap-2 font-mono text-xs">
                    <span className="w-9 text-right" style={{ color: masked ? 'var(--ink-faint)' : 'var(--ink)' }}>{t}</span>
                    <div className="h-3.5 flex-1 overflow-hidden rounded-sm bg-[color-mix(in_srgb,var(--rule)_40%,transparent)]">
                      <div
                        className="h-full"
                        style={{
                          width: `${(stage >= 2 ? row[j] : 0) * 100}%`,
                          transition: 'width 0.35s ease',
                          background: j > q ? 'var(--red)' : 'var(--violet)',
                        }}
                      />
                    </div>
                    <span className="w-10 text-right text-ink-soft">{stage >= 2 ? row[j].toFixed(2) : '·'}</span>
                    <span className="w-20 whitespace-nowrap text-[0.65rem]" style={{ color: masked ? 'var(--ink-faint)' : 'var(--green)' }}>
                      {masked ? 'future' : j < q ? 'K,V cached' : 'K,V new'}
                    </span>
                  </div>
                );
              })}
            </div>

            {stage === 3 && (
              <p className="mt-3 font-mono text-xs text-ink">
                out = {row.map((w, j) => (w > 0.005 ? `${w.toFixed(2)}·v${j}` : null)).filter(Boolean).join(' + ')}
              </p>
            )}
            {leaks ? (
              <p className="mt-3 text-sm" style={{ color: 'var(--red)' }}>
                {(leaks * 100).toFixed(0)}% of this token's attention goes to the future. Turn the mask back on.
              </p>
            ) : (
              <p className="mt-3 text-sm text-ink-soft">
                To compute this row, the model needs K and V for positions 0–{q} only. Rows 0–{Math.max(0, q - 1)} were already computed in
                earlier steps.
              </p>
            )}
          </div>
        </div>
      </div>
    </MotionConfig>
  );
}
