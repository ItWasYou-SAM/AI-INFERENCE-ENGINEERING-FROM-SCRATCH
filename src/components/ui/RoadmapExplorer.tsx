import { useEffect, useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'motion/react';
import type { Level } from '../../data/roadmap';

const PROGRESS_KEY = 'inference-progress-v1';
const LAB = {
  cpu: { c: 'var(--green)', t: 'CPU' },
  'gpu-optional': { c: 'var(--amber)', t: 'GPU opt.' },
  'gpu-required': { c: 'var(--red)', t: 'GPU' },
} as const;

export default function RoadmapExplorer({ levels, written }: { levels: Level[]; written: string[] }) {
  const [open, setOpen] = useState<string | null>('m00');
  const [done, setDone] = useState<Set<string>>(new Set());
  const writtenSet = new Set(written);

  useEffect(() => {
    const load = () => {
      try {
        setDone(new Set(JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? '[]')));
      } catch {
        /* ignore */
      }
    };
    load();
    window.addEventListener('progresschange', load);
    return () => window.removeEventListener('progresschange', load);
  }, []);

  const total = levels.reduce((n, l) => n + l.modules.reduce((k, m) => k + m.chapters.length, 0), 0);

  return (
    <MotionConfig reducedMotion="user">
    <div>
      <div className="mb-10 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-xs text-ink-soft">
        <span>
          <b className="text-ink">{done.size}</b> / {total} chapters complete
        </span>
        <span className="h-1.5 w-48 overflow-hidden rounded-full bg-paper-3">
          <motion.span
            className="block h-full bg-green"
            initial={{ width: 0 }}
            animate={{ width: `${(done.size / total) * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </span>
        <span className="flex gap-3">
          {Object.values(LAB).map((l) => (
            <span key={l.t} className="flex items-center gap-1">
              <i className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: l.c }} /> {l.t}
            </span>
          ))}
          <span className="text-blue">✦ interactive</span>
        </span>
      </div>

      <ol className="relative">
        {/* the ink spine */}
        <motion.span
          aria-hidden
          className="absolute top-2 bottom-2 left-[1.15rem] w-px origin-top border-l-2 border-dashed border-rule"
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ duration: 1.4, ease: 'easeInOut' }}
        />
        {levels.map((level, li) => (
          <li key={level.num} className="relative mb-14 pl-14">
            <motion.span
              initial={{ scale: 0 }}
              whileInView={{ scale: 1 }}
              viewport={{ once: true }}
              transition={{ type: 'spring', stiffness: 300, damping: 18, delay: 0.1 }}
              className="absolute top-0 left-0 grid h-9 w-9 place-items-center rounded-full border-2 border-ink bg-paper font-display text-lg"
            >
              {level.num}
            </motion.span>
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.6, delay: li * 0.05 }}
            >
              <p className="eyebrow">Level {level.num}</p>
              <h2 className="mt-1 text-3xl">
                {level.name} <span className="text-ink-faint italic">— {level.question}</span>
              </h2>
            </motion.div>

            <div className="mt-6 grid gap-3">
              {level.modules.map((m, mi) => {
                const isOpen = open === m.id;
                const keys = m.chapters.map((c) => `${m.id}/${c.slug}`);
                const nDone = keys.filter((k) => done.has(k)).length;
                return (
                  <motion.div
                    key={m.id}
                    layout
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true, margin: '-40px' }}
                    transition={{ duration: 0.5, delay: mi * 0.06 }}
                    className="sheet overflow-hidden"
                  >
                    <button
                      type="button"
                      aria-expanded={isOpen}
                      onClick={() => setOpen(isOpen ? null : m.id)}
                      className="flex w-full items-start gap-4 p-4 text-left sm:p-5"
                    >
                      <span className="mt-0.5 w-10 shrink-0 font-mono text-sm text-ink-faint">{m.id === 'capstone' ? 'CAP' : `M${m.num}`}</span>
                      <span className="min-w-0 flex-1">
                        <span className="block font-display text-xl leading-snug">{m.title}</span>
                        <span className="mt-1 block text-sm text-ink-soft">{m.tagline}</span>
                        <span className="mt-2 flex flex-wrap gap-1.5">
                          {m.sources.map((s) => (
                            <span key={s.id} className="rounded border border-rule px-1.5 py-px font-mono text-[0.62rem] text-ink-soft" title={s.note}>
                              {s.id}
                            </span>
                          ))}
                        </span>
                      </span>
                      <span className="flex shrink-0 flex-col items-end gap-2">
                        <span className="font-mono text-xs text-ink-faint">
                          {nDone > 0 ? `${nDone}/` : ''}
                          {m.chapters.length} ch
                        </span>
                        <motion.span animate={{ rotate: isOpen ? 45 : 0 }} className="text-xl leading-none text-ink-soft">
                          +
                        </motion.span>
                      </span>
                    </button>
                    <AnimatePresence initial={false}>
                      {isOpen && (
                        <motion.ul
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.35, ease: [0.2, 0.7, 0.2, 1] }}
                          className="overflow-hidden border-t border-dashed border-rule"
                        >
                          {m.chapters.map((ch, ci) => {
                            const key = `${m.id}/${ch.slug}`;
                            const isDone = done.has(key);
                            const isWritten = writtenSet.has(key);
                            return (
                              <motion.li
                                key={key}
                                initial={{ opacity: 0, x: -8 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: ci * 0.03 }}
                              >
                                <a
                                  href={`/learn/${key}`}
                                  className="group flex items-center gap-3 px-5 py-2.5 transition-colors hover:bg-paper-2"
                                >
                                  <span
                                    className={`grid h-5 w-5 shrink-0 place-items-center rounded-full border text-[0.6rem] ${
                                      isDone ? 'border-green bg-green text-paper' : 'border-rule'
                                    }`}
                                  >
                                    {isDone ? '✓' : ''}
                                  </span>
                                  <span className="w-10 shrink-0 font-mono text-xs text-ink-faint">{ch.id}</span>
                                  <span className={`flex-1 ${isWritten ? '' : 'text-ink-soft'}`}>
                                    <span className="group-hover:underline">{ch.title}</span>
                                    {isWritten && (
                                      <span className="ml-2 rounded px-1 font-mono text-[0.6rem] tracking-wide text-ink uppercase" style={{ background: 'var(--highlight)' }}>
                                        ready
                                      </span>
                                    )}
                                  </span>
                                  {ch.interactive && <span className="text-xs text-blue" title="Interactive">✦</span>}
                                  {ch.lab && (
                                    <span className="flex items-center gap-1 font-mono text-[0.62rem]" style={{ color: LAB[ch.lab].c }}>
                                      <i className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: LAB[ch.lab].c }} />
                                      {LAB[ch.lab].t}
                                    </span>
                                  )}
                                </a>
                              </motion.li>
                            );
                          })}
                        </motion.ul>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </div>
          </li>
        ))}
      </ol>
    </div>
    </MotionConfig>
  );
}
