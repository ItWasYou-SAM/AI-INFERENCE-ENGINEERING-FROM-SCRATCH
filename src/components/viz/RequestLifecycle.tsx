import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'motion/react';

/* An illustrative (not measured) timeline for one chat request, in simulated milliseconds. */
const PROMPT = ['Explain', ' the', ' KV', ' cache', ' in', ' one', ' sentence', '.'];
const OUTPUT = [' It', ' stores', ' past', ' keys', ' and', ' values', ' so', ' attention', ' never', ' recomputes', ' them', '.', '<eos>'];
const IDS = [849, 279, 85361, 6636, 304, 832, 11914, 13];

const T = {
  network: [0, 15],
  tokenize: [15, 18],
  queue: [18, 45],
  prefill: [45, 165],
  decodeStep: 30,
};
const firstTokenAt = T.prefill[1];
const tokenTime = (i: number) => firstTokenAt + i * T.decodeStep; // i-th output token (0 = first)
const END = tokenTime(OUTPUT.length - 1) + 6;

const STAGES = [
  {
    id: 'network',
    name: 'Request',
    span: T.network,
    text: 'The client POSTs JSON to an OpenAI-compatible endpoint (/v1/chat/completions) with the messages, max_tokens, temperature and stream: true. The API server validates it and applies the chat template.',
  },
  {
    id: 'tokenize',
    name: 'Tokenize',
    span: T.tokenize,
    text: 'The tokenizer turns text into integer token IDs. The model never sees characters, only IDs. This runs on the CPU and is usually cheap.',
  },
  {
    id: 'queue',
    name: 'Schedule',
    span: T.queue,
    text: 'The request waits in the scheduler’s queue until there is room in the running batch and enough free KV-cache blocks for its prompt. Under load, this wait can dominate TTFT.',
  },
  {
    id: 'prefill',
    name: 'Prefill',
    span: T.prefill,
    text: 'All 8 prompt tokens go through the model in one forward pass. Their keys and values are written into the KV cache, and the last position’s logits are sampled to produce the first output token. This phase is compute-heavy.',
  },
  {
    id: 'decode',
    name: 'Decode',
    span: [firstTokenAt, END - 6],
    text: 'Each step feeds only the newest token, reads every cached K/V to attend over the whole history, appends one new K/V pair, and samples the next token. One full read of the weights produces one token. This phase is memory-bound.',
  },
  {
    id: 'stream',
    name: 'Stream & finish',
    span: [END - 6, END],
    text: 'Each token is detokenized and streamed back as a server-sent event while decode continues. On <eos> (or max_tokens) the request finishes and its KV-cache blocks are freed for the next request.',
  },
] as const;

function stageAt(t: number) {
  if (t >= END - 6) return 5;
  if (t >= firstTokenAt) return 4;
  for (let i = 3; i >= 0; i--) if (t >= STAGES[i].span[0]) return i;
  return 0;
}

export default function RequestLifecycle() {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.12);
  const raf = useRef<number | null>(null);
  const last = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) return;
    const tick = (now: number) => {
      if (last.current != null) {
        const dt = (now - last.current) * speed;
        setT((x) => {
          const nx = Math.min(END, x + dt);
          if (nx >= END) setPlaying(false);
          return nx;
        });
      }
      last.current = now;
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
      last.current = null;
    };
  }, [playing, speed]);

  const stage = stageAt(t);
  const emitted = OUTPUT.filter((_, i) => t >= tokenTime(i)).length;
  const prefillFrac = Math.max(0, Math.min(1, (t - T.prefill[0]) / (T.prefill[1] - T.prefill[0])));
  const promptKV = t >= T.prefill[1] ? PROMPT.length : Math.floor(prefillFrac * PROMPT.length);
  // a generated token's K/V is written when it is fed back in on the next step
  const genKV = Math.max(0, emitted - 1);
  const tokenized = t >= T.tokenize[1];

  const step = () => {
    setPlaying(false);
    // jump to the next interesting boundary: stage start, or the next token during decode
    const marks = [...STAGES.map((s) => s.span[0]), ...OUTPUT.map((_, i) => tokenTime(i)), END].sort((a, b) => a - b);
    const next = marks.find((m) => m > t + 0.01);
    setT(next ?? END);
  };
  const reset = () => {
    setPlaying(false);
    setT(0);
  };

  const pct = (ms: number) => `${(ms / END) * 100}%`;

  return (
    <MotionConfig reducedMotion="user">
      <div className="not-prose text-ink">
        {/* controls */}
        <div className="mb-5 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => (t >= END ? (setT(0), setPlaying(true)) : setPlaying((p) => !p))}
            className="rounded-full bg-ink px-4 py-1.5 font-mono text-xs tracking-wide text-paper uppercase"
          >
            {playing ? '❚❚ Pause' : t >= END ? '↺ Replay' : '▶ Play'}
          </button>
          <button type="button" onClick={step} className="rounded-full border border-ink px-3 py-1.5 font-mono text-xs tracking-wide uppercase">
            Step →
          </button>
          <button type="button" onClick={reset} className="rounded-full border border-rule px-3 py-1.5 font-mono text-xs tracking-wide text-ink-soft uppercase">
            Reset
          </button>
          <span className="ml-auto flex items-center gap-1 font-mono text-[0.68rem] text-ink-faint">
            speed
            {[0.05, 0.12, 0.3].map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSpeed(s)}
                className={`rounded px-1.5 py-0.5 ${speed === s ? 'bg-ink text-paper' : 'hover:text-ink'}`}
              >
                {s === 0.05 ? 'slow' : s === 0.12 ? 'normal' : 'fast'}
              </button>
            ))}
          </span>
        </div>

        {/* stage pipeline */}
        <ol className="grid grid-cols-3 gap-1.5 sm:grid-cols-6">
          {STAGES.map((s, i) => {
            const state = i < stage ? 'done' : i === stage ? 'now' : 'todo';
            return (
              <li
                key={s.id}
                className="relative rounded-md border px-2 py-2 text-center transition-colors duration-300"
                style={{
                  borderColor: state === 'now' ? 'var(--blue)' : 'var(--rule)',
                  background: state === 'now' ? 'color-mix(in srgb, var(--blue) 12%, var(--paper))' : 'var(--paper)',
                  opacity: state === 'todo' ? 0.55 : 1,
                }}
              >
                <span className="block font-mono text-[0.6rem] text-ink-faint">{state === 'done' ? '✓' : i + 1}</span>
                <span className="block font-display text-sm leading-tight">{s.name}</span>
                {state === 'now' && playing && (
                  <motion.span
                    className="absolute inset-x-2 bottom-1 h-0.5 rounded-full bg-blue"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                  />
                )}
              </li>
            );
          })}
        </ol>

        {/* text & tokens */}
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <div className="rounded-md border border-dashed border-rule bg-paper p-4">
            <p className="eyebrow mb-2">Prompt → token IDs (illustrative)</p>
            <div className="flex flex-wrap gap-1">
              {PROMPT.map((tok, i) => (
                <span key={i} className="inline-flex flex-col items-center rounded border border-rule bg-paper-2 px-1.5 py-0.5">
                  <span className="font-mono text-xs whitespace-pre">{tok}</span>
                  <AnimatePresence>
                    {tokenized && (
                      <motion.span
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.03 }}
                        className="font-mono text-[0.6rem] text-blue"
                      >
                        {IDS[i]}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </span>
              ))}
            </div>
            <p className="eyebrow mt-4 mb-2">Streamed output</p>
            <p className="min-h-[3.2rem] font-serif text-[0.95rem] leading-relaxed">
              {OUTPUT.slice(0, emitted)
                .filter((x) => x !== '<eos>')
                .map((tok, i) => (
                  <motion.span key={i} initial={{ opacity: 0, backgroundColor: 'var(--highlight)' }} animate={{ opacity: 1, backgroundColor: 'rgba(0,0,0,0)' }} transition={{ duration: 0.6 }}>
                    {tok}
                  </motion.span>
                ))}
              {emitted < OUTPUT.length && stage >= 3 && <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-ink" />}
              {emitted === OUTPUT.length && <span className="ml-2 font-mono text-[0.65rem] text-ink-faint">[eos]</span>}
            </p>
          </div>

          <div className="rounded-md border border-dashed border-rule bg-paper p-4">
            <p className="eyebrow mb-2 flex justify-between">
              <span>KV cache (one slot per token)</span>
              <span className="text-ink">{stage === 5 && t >= END ? 'freed' : `${promptKV + genKV} tokens`}</span>
            </p>
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: PROMPT.length + OUTPUT.length - 1 }, (_, i) => {
                const isPrompt = i < PROMPT.length;
                const filled = t >= END ? false : isPrompt ? i < promptKV : i - PROMPT.length < genKV;
                return (
                  <motion.span
                    key={i}
                    animate={{
                      backgroundColor: filled ? (isPrompt ? 'var(--ink-soft)' : 'var(--blue)') : 'rgba(0,0,0,0)',
                      scale: filled ? 1 : 0.85,
                    }}
                    transition={{ duration: 0.25 }}
                    className="h-6 w-6 rounded-sm border border-rule"
                    title={isPrompt ? `prompt token ${i}` : `generated token ${i - PROMPT.length}`}
                  />
                );
              })}
            </div>
            <p className="mt-2 font-mono text-[0.65rem] text-ink-faint">
              <i className="mr-1 inline-block h-2 w-2 bg-ink-soft" />
              prompt (written in prefill) <i className="mr-1 ml-3 inline-block h-2 w-2 bg-blue" />
              generated (one per decode step)
            </p>
          </div>
        </div>

        {/* explanation */}
        <AnimatePresence mode="wait">
          <motion.p
            key={stage}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-4 min-h-[4.5rem] text-[0.95rem] leading-relaxed"
          >
            <b className="font-display">{STAGES[stage].name}. </b>
            {STAGES[stage].text}
          </motion.p>
        </AnimatePresence>

        {/* timeline */}
        <div className="mt-4">
          <div className="relative h-10">
            {STAGES.slice(0, 4).map((s, i) => (
              <div
                key={s.id}
                className="absolute top-3 h-4 rounded-sm"
                style={{
                  left: pct(s.span[0]),
                  width: pct(s.span[1] - s.span[0]),
                  background: i === 3 ? 'var(--ink-soft)' : 'var(--rule)',
                }}
                title={s.name}
              />
            ))}
            {OUTPUT.map((_, i) => (
              <div key={i} className="absolute top-2 h-6 w-[2px] rounded" style={{ left: pct(tokenTime(i)), background: t >= tokenTime(i) ? 'var(--blue)' : 'var(--rule)' }} />
            ))}
            {/* playhead */}
            <div className="absolute top-0 h-10 w-[2px] bg-red" style={{ left: pct(t) }} />
            {/* TTFT bracket */}
            <div className="absolute top-8 border-t-2 border-dashed border-red/60" style={{ left: 0, width: pct(firstTokenAt) }} />
          </div>
          <div className="mt-1 flex flex-wrap justify-between gap-2 font-mono text-[0.68rem] text-ink-soft">
            <span>
              <b className="text-red">TTFT</b> = {firstTokenAt} ms (network + tokenize + queue + prefill)
            </span>
            <span>
              <b className="text-blue">TPOT</b> = {T.decodeStep} ms per token
            </span>
            <span>t = {Math.round(t)} ms</span>
          </div>
          <p className="mt-1 font-mono text-[0.62rem] text-ink-faint">Timings are illustrative, chosen to show the shape of a request, not measured.</p>
        </div>
      </div>
    </MotionConfig>
  );
}
