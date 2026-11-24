import { useState } from 'react';
import { motion, MotionConfig } from 'motion/react';

const MYTHS = [
  {
    myth: 'A GPU with 2× the TFLOPS writes text 2× faster.',
    fact: 'Usually not. Single-user decode is memory-bound, so speed follows memory bandwidth (TB/s), not TFLOPS.',
    href: '/learn/m03/roofline',
    ch: '3.3',
  },
  {
    myth: 'Longer prompts make every output token slower.',
    fact: 'Mostly they delay the first token (prefill). Per-token speed barely moves; only the larger KV cache adds a little.',
    href: '/learn/m00/life-of-a-request',
    ch: '0.2',
  },
  {
    myth: 'Batching users together makes everyone slower.',
    fact: 'Until you reach the ridge point, extra users ride along on the same weight read. Throughput scales almost linearly.',
    href: '/learn/m03/roofline',
    ch: '3.3',
  },
  {
    myth: 'The model weights are what fill GPU memory.',
    fact: 'At long context the KV cache wins: one 128K-token Mistral 7B user needs 17 GB, more than the 14.5 GB of weights.',
    href: '/learn/m09/kv-cache-size',
    ch: '9.1',
  },
  {
    myth: 'Quantization only saves memory.',
    fact: 'In decode, fewer bytes per weight means fewer bytes moved per token, so it is also a direct speedup.',
    href: '/learn/m12/number-formats',
    ch: '12.1',
  },
  {
    myth: 'With enough replicas, routing doesn’t matter.',
    fact: 'Sending a conversation back to the replica that already holds its KV cache skips re-running the whole prefix.',
    href: '/learn/m15/load-balancing-policies',
    ch: '15.3',
  },
];


function Card({ m, i }: { m: (typeof MYTHS)[number]; i: number }) {
  const [flipped, setFlipped] = useState(false);
  return (
    <motion.div
      role="button"
      tabIndex={0}
      onClick={() => setFlipped((f) => !f)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          setFlipped((f) => !f);
        }
      }}
      aria-pressed={flipped}
      aria-label={flipped ? `Fact: ${m.fact}` : `Myth: ${m.myth}. Click to reveal the fact.`}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ delay: i * 0.07, duration: 0.55 }}
      whileHover={{ y: -4 }}
      className="group relative h-56 w-full cursor-pointer text-left [perspective:1200px]"
    >
      <motion.div
        className="relative h-full w-full [transform-style:preserve-3d]"
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ type: 'spring', stiffness: 160, damping: 20 }}
      >
        {/* front: myth */}
        <div className="sheet absolute inset-0 flex flex-col p-5 [backface-visibility:hidden]">
          <span className="eyebrow">
            Myth #{i + 1}
          </span>
          <p className="mt-3 font-display text-xl leading-snug">“{m.myth}”</p>
          <span className="mt-auto flex items-center gap-2 font-mono text-[0.68rem] tracking-wide text-ink-faint uppercase transition-colors group-hover:text-ink">
            <motion.span animate={{ rotate: [0, 180, 360] }} transition={{ repeat: Infinity, duration: 4, ease: 'easeInOut' }}>
              ↻
            </motion.span>
            Tap to flip
          </span>
        </div>
        {/* back: fact */}
        <div
          className="absolute inset-0 flex flex-col rounded-[10px] bg-ink p-5 text-paper [backface-visibility:hidden] [transform:rotateY(180deg)]"
        >
          <span className="font-mono text-[0.68rem] tracking-[0.12em] text-blue uppercase">Actually…</span>
          <p className="mt-3 text-[0.98rem] leading-snug">{m.fact}</p>
          <a
            href={m.href}
            onClick={(e) => e.stopPropagation()}
            className="mt-auto inline-flex w-fit items-center gap-1 rounded-full border border-paper/30 px-3 py-1 font-mono text-[0.68rem] tracking-wide uppercase hover:bg-paper/10"
          >
            Learn why · {m.ch} →
          </a>
        </div>
      </motion.div>
    </motion.div>
  );
}

export default function MythCards() {
  return (
    <MotionConfig reducedMotion="user">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {MYTHS.map((m, i) => (
          <Card key={i} m={m} i={i} />
        ))}
      </div>
    </MotionConfig>
  );
}
