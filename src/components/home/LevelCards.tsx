import { useRef } from 'react';
import { motion, MotionConfig, useMotionValue, useSpring, useTransform } from 'motion/react';

interface LevelCard {
  num: number;
  name: string;
  question: string;
  href: string;
  chapters: number;
  modules: { label: string; title: string }[];
}

function Tilt({ children, i }: { children: React.ReactNode; i: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0.5);
  const my = useMotionValue(0.5);
  const rx = useSpring(useTransform(my, [0, 1], [6, -6]), { stiffness: 200, damping: 18 });
  const ry = useSpring(useTransform(mx, [0, 1], [-7, 7]), { stiffness: 200, damping: 18 });
  const glowX = useTransform(mx, (v) => `${v * 100}%`);
  const glowY = useTransform(my, (v) => `${v * 100}%`);
  const glow = useTransform([glowX, glowY], ([x, y]) => `radial-gradient(240px circle at ${x} ${y}, color-mix(in srgb, var(--blue) 8%, transparent), transparent 70%)`);
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ delay: i * 0.1, duration: 0.6 }}
      onPointerMove={(e) => {
        const r = ref.current!.getBoundingClientRect();
        mx.set((e.clientX - r.left) / r.width);
        my.set((e.clientY - r.top) / r.height);
      }}
      onPointerLeave={() => {
        mx.set(0.5);
        my.set(0.5);
      }}
      style={{ rotateX: rx, rotateY: ry, transformPerspective: 900 }}
      className="relative h-full"
    >
      <motion.span
        aria-hidden
        className="pointer-events-none absolute inset-0 z-10 rounded-[10px] opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{ background: glow }}
      />
      {children}
    </motion.div>
  );
}

export default function LevelCards({ levels }: { levels: LevelCard[] }) {
  return (
    <MotionConfig reducedMotion="user">
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {levels.map((l, i) => (
          <a key={l.num} href={l.href} className="group block h-full">
            <Tilt i={i}>
              <div className="sheet relative flex h-full flex-col overflow-hidden p-5">
                <div className="flex items-baseline justify-between">
                  <span className="font-display text-6xl leading-none text-ink-faint transition-colors duration-300 group-hover:text-blue">
                    {l.num}
                  </span>
                  <span className="font-mono text-[0.65rem] text-ink-faint">{l.chapters} chapters</span>
                </div>
                <span className="eyebrow mt-3">{l.name}</span>
                <span className="mt-1 font-display text-lg leading-snug">{l.question}</span>
                <ul className="mt-4 space-y-1.5 text-sm text-ink-soft">
                  {l.modules.map((m) => (
                    <li key={m.label} className="flex gap-2">
                      <span className="w-8 shrink-0 font-mono text-[0.68rem] text-ink-faint">{m.label}</span>
                      <span>{m.title}</span>
                    </li>
                  ))}
                </ul>
                <span className="mt-auto pt-5 font-mono text-[0.68rem] tracking-wide text-ink-faint uppercase transition-colors group-hover:text-ink">
                  Start level {l.num}
                  <span className="ml-1 inline-block transition-transform duration-300 group-hover:translate-x-1">→</span>
                </span>
              </div>
            </Tilt>
          </a>
        ))}
      </div>
    </MotionConfig>
  );
}
