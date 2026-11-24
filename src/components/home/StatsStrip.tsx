import { useEffect, useRef } from 'react';
import { animate, motion, MotionConfig, useInView, useMotionValue, useTransform } from 'motion/react';

const FACTS = [
  { value: 0.3, digits: 1, unit: '%', label: 'of an H100’s compute is used when it decodes for one user', href: '/learn/m03/roofline' },
  { value: 131, digits: 0, unit: ' KB', label: 'of KV cache for every token Mistral 7B remembers', href: '/learn/m09/kv-cache-size' },
  { value: 4.3, digits: 1, unit: ' ms', label: 'minimum per token for a 7B model on an H100: just moving the weights', href: '/learn/m00/what-is-inference-engineering' },
  { value: 107, digits: 0, unit: '', label: 'users fit on one 80 GB GPU at 4K context, and only 26 at 16K', href: '/learn/m09/kv-cache-size' },
];

function CountUp({ to, digits, start }: { to: number; digits: number; start: boolean }) {
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => v.toFixed(digits));
  useEffect(() => {
    if (!start) return;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      mv.set(to);
      return;
    }
    const c = animate(mv, to, { duration: 1.6, ease: [0.16, 1, 0.3, 1] });
    return c.stop;
  }, [start, to, mv]);
  return <motion.span>{text}</motion.span>;
}

export default function StatsStrip() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  return (
    <MotionConfig reducedMotion="user">
      <div ref={ref} className="grid gap-px overflow-hidden rounded-2xl border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
        {FACTS.map((f, i) => (
          <motion.a
            key={i}
            href={f.href}
            initial={{ opacity: 0, y: 16 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: i * 0.12, duration: 0.6 }}
            className="group relative block bg-paper p-6 transition-colors hover:bg-paper-2"
          >
            <span className="absolute inset-x-0 bottom-0 h-0.5 origin-left scale-x-0 bg-blue transition-transform duration-500 group-hover:scale-x-100" />
            <p className="font-display text-5xl tabular-nums text-ink transition-colors group-hover:text-blue">
              <CountUp to={f.value} digits={f.digits} start={inView} />
              <span className="text-2xl text-ink-faint">{f.unit}</span>
            </p>
            <p className="mt-2 text-sm leading-snug text-ink-soft">{f.label}</p>
            <p className="mt-3 font-mono text-[0.65rem] tracking-wide text-ink-faint uppercase transition-colors group-hover:text-ink">See why →</p>
          </motion.a>
        ))}
      </div>
    </MotionConfig>
  );
}
