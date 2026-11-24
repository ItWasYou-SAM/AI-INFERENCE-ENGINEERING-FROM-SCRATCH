import { useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'motion/react';

export interface QuizQuestion {
  q: string;
  options: string[];
  answer: number;
  explain: string;
}

export default function Quiz({ questions, title = 'Check your understanding' }: { questions: QuizQuestion[]; title?: string }) {
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null));
  const score = picked.filter((p, i) => p === questions[i].answer).length;
  const done = picked.every((p) => p !== null);

  return (
    <MotionConfig reducedMotion="user">
    <section className="sheet not-prose my-10 p-5 sm:p-7" aria-label={title}>
      <div className="mb-5 flex items-baseline justify-between gap-4">
        <h3 className="font-display text-xl">{title}</h3>
        <span className="eyebrow">
          {done ? `${score} / ${questions.length} correct` : `${picked.filter((p) => p !== null).length} / ${questions.length} answered`}
        </span>
      </div>
      <ol className="space-y-7">
        {questions.map((q, qi) => {
          const sel = picked[qi];
          return (
            <li key={qi}>
              <p className="mb-3 leading-snug">
                <span className="mr-2 font-mono text-sm text-ink-faint">{qi + 1}.</span>
                {q.q}
              </p>
              <div className="grid gap-2">
                {q.options.map((opt, oi) => {
                  const isPicked = sel === oi;
                  const isRight = oi === q.answer;
                  const reveal = sel !== null;
                  let tone = 'border-rule hover:border-ink-soft';
                  if (reveal && isRight) tone = 'border-green bg-green/10';
                  else if (reveal && isPicked) tone = 'border-red bg-red/10';
                  return (
                    <motion.button
                      key={oi}
                      type="button"
                      disabled={reveal}
                      whileTap={{ scale: 0.98 }}
                      animate={reveal && isPicked && !isRight ? { x: [0, -6, 6, -3, 0] } : {}}
                      transition={{ duration: 0.35 }}
                      onClick={() => setPicked((p) => p.map((v, i) => (i === qi ? oi : v)))}
                      className={`flex items-start gap-3 rounded-md border border-dashed px-3 py-2 text-left text-[0.95rem] transition-colors disabled:cursor-default ${tone}`}
                    >
                      <span className="mt-0.5 font-mono text-xs text-ink-faint">{String.fromCharCode(65 + oi)}</span>
                      <span>{opt}</span>
                    </motion.button>
                  );
                })}
              </div>
              <AnimatePresence>
                {sel !== null && (
                  <motion.p
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="overflow-hidden text-sm text-ink-soft"
                  >
                    <span className="mt-3 block">
                      <strong className={sel === q.answer ? 'text-green' : 'text-red'}>
                        {sel === q.answer ? 'Correct. ' : 'Not quite. '}
                      </strong>
                      {q.explain}
                    </span>
                  </motion.p>
                )}
              </AnimatePresence>
            </li>
          );
        })}
      </ol>
      {done && (
        <button
          type="button"
          className="mt-6 font-mono text-xs tracking-wide text-blue uppercase hover:underline"
          onClick={() => setPicked(questions.map(() => null))}
        >
          ↺ Reset quiz
        </button>
      )}
    </section>
    </MotionConfig>
  );
}
