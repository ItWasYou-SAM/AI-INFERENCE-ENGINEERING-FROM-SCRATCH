import { useEffect, useState } from 'react';
import { motion, MotionConfig } from 'motion/react';

const PROMPT = ['Why', 'is', 'decode', 'slow', '?'];
const OUTPUT = ['It', 'reads', 'every', 'weight', 'for', 'each', 'new', 'token', '.'];
const ALL = [...PROMPT, ...OUTPUT];
const P = PROMPT.length;
const MAX_STEP = OUTPUT.length; // step 0 = prefill, steps 1..MAX = decode

/** Rows of K/V computed at step s (s = 0 is prefill over the prompt). */
const rowsAtStep = (s: number) => P + s; // sequence length seen at step s
const naiveTotal = (s: number) => Array.from({ length: s + 1 }, (_, i) => rowsAtStep(i)).reduce((a, b) => a + b, 0);
const cachedTotal = (s: number) => P + s;

function Grid({ step, cached }: { step: number; cached: boolean }) {
  const len = rowsAtStep(step);
  return (
    <div className="flex flex-wrap gap-1" aria-hidden="true">
      {ALL.map((t, i) => {
        const exists = i < len;
        const isNew = cached ? (step === 0 ? i < P : i === len - 1) : exists;
        const fromCache = cached && exists && !isNew;
        const color = !exists ? 'transparent' : isNew ? (cached ? 'var(--green)' : 'var(--red)') : 'var(--ink-faint)';
        return (
          <motion.div
            key={i}
            className="flex h-12 w-12 flex-col items-center justify-center rounded border font-mono text-[0.6rem]"
            style={{ borderColor: exists ? color : 'var(--rule)', borderStyle: exists ? 'solid' : 'dashed' }}
            animate={{
              backgroundColor: exists ? `color-mix(in srgb, ${color} ${isNew ? 28 : 12}%, transparent)` : 'rgba(0,0,0,0)',
              scale: isNew && exists ? [0.85, 1] : 1,
            }}
            transition={{ duration: 0.35 }}
          >
            <span className={exists ? 'text-ink' : 'text-ink-faint'}>{t}</span>
            <span style={{ color: exists ? color : 'var(--ink-faint)' }}>
              {!exists ? '' : fromCache ? 'cache' : 'K V'}
            </span>
          </motion.div>
        );
      })}
    </div>
  );
}

export default function KVCacheCompare() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (step >= MAX_STEP) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setStep((s) => s + 1), 750);
    return () => clearTimeout(t);
  }, [playing, step]);

  const naive = naiveTotal(step);
  const cached = cachedTotal(step);
  const maxNaive = naiveTotal(MAX_STEP);

  return (
    <MotionConfig reducedMotion="user">
      <div className="not-prose text-ink">
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => {
              if (step >= MAX_STEP) setStep(0);
              setPlaying((p) => !p);
            }}
            className="rounded-md border border-ink bg-ink px-3 py-1 font-mono text-xs text-paper"
          >
            {playing ? 'Pause' : step >= MAX_STEP ? 'Replay' : 'Play'}
          </button>
          <button type="button" disabled={step >= MAX_STEP} onClick={() => { setPlaying(false); setStep((s) => Math.min(MAX_STEP, s + 1)); }}
            className="rounded-md border border-rule px-3 py-1 font-mono text-xs text-ink-soft hover:text-ink disabled:opacity-40">
            Step
          </button>
          <button type="button" onClick={() => { setPlaying(false); setStep(0); }}
            className="rounded-md border border-rule px-3 py-1 font-mono text-xs text-ink-soft hover:text-ink">
            Reset
          </button>
          <span className="ml-auto font-mono text-[0.7rem] text-ink-faint">
            {step === 0 ? 'step 0 · prefill the prompt' : `step ${step} · decode token “${OUTPUT[step - 1]}”`} · sequence length {rowsAtStep(step)}
          </span>
        </div>

        <div className="space-y-5">
          <section>
            <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4">
              <p className="eyebrow" style={{ color: 'var(--red)' }}>Without a cache: recompute every position, every step</p>
              <span className="font-mono text-xs text-ink-soft">this step: {rowsAtStep(step)} rows</span>
            </div>
            <Grid step={step} cached={false} />
          </section>
          <section>
            <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4">
              <p className="eyebrow" style={{ color: 'var(--green)' }}>With a KV cache: compute only the new position</p>
              <span className="font-mono text-xs text-ink-soft">this step: {step === 0 ? P : 1} row{step === 0 ? 's' : ''}</span>
            </div>
            <Grid step={step} cached />
          </section>
        </div>

        <div className="mt-6 rounded-md border border-rule bg-paper p-4">
          <p className="eyebrow">Total K/V rows computed so far (one row = one token through every layer)</p>
          {[
            { label: 'no cache', v: naive, color: 'var(--red)' },
            { label: 'KV cache', v: cached, color: 'var(--green)' },
          ].map((b) => (
            <div key={b.label} className="mt-2 flex items-center gap-3 font-mono text-xs">
              <span className="w-16">{b.label}</span>
              <div className="h-4 flex-1 overflow-hidden rounded-sm bg-[color-mix(in_srgb,var(--rule)_40%,transparent)]">
                <div className="h-full" style={{ width: `${(b.v / maxNaive) * 100}%`, transition: 'width 0.4s ease',  background: b.color }} />
              </div>
              <span className="w-10 text-right text-ink">{b.v}</span>
            </div>
          ))}
          <p className="mt-3 text-sm text-ink-soft">
            {step === 0
              ? 'Prefill costs the same either way: every prompt token has to go through the model once.'
              : <>After {step} decode step{step > 1 ? 's' : ''} the uncached loop has done <b className="text-ink">{(naive / cached).toFixed(1)}×</b> the
                work. The gap grows with every token, because uncached cost per step grows with length while cached cost stays at one row.</>}
          </p>
        </div>
      </div>
    </MotionConfig>
  );
}
