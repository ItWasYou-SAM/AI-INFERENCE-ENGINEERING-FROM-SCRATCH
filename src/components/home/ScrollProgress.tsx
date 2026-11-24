import { motion, useScroll, useSpring } from 'motion/react';

/** Thin gradient bar under the header that fills as you scroll the page. */
export default function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 140, damping: 26, restDelta: 0.001 });
  return <motion.div aria-hidden className="fixed top-14 right-0 left-0 z-50 h-[3px] origin-left" style={{ scaleX, background: 'var(--grad)' }} />;
}
