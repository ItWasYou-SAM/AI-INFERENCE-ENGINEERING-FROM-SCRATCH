import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * fig. 0: one decode step, on loop.
 *   ① weights stream from HBM into the chip
 *   ② the SMs light up as the math sweeps through the layers
 *   ③ exactly one token comes out, then it starts again
 * Calm, deterministic motion: no random particles, no camera spin.
 */

const COLS = 10;
const ROWS = 6;
const SM = 0.5;
const GAP = 0.12;
const DIE_W = COLS * SM + (COLS - 1) * GAP;
const DIE_D = ROWS * SM + (ROWS - 1) * GAP;
const STACK_X = DIE_W / 2 + 1.75;
const STACK_Z = [-0.95, 0.95];
const PACKETS_PER_LANE = 7;
const FOV = 36;
const ELEV = THREE.MathUtils.degToRad(52);
const BOARD_W = 2 * (STACK_X + 0.8);
const HALF_W = BOARD_W / 2 + 0.15; // half-width of what must stay in view
const HALF_D = DIE_D / 2 + 1.6;

const CYCLE = 4.2; // seconds per decode step (slowed down to be readable)
const A_END = 0.58; // ① stream weights
const B_END = 0.74; // ② all SMs busy
// ③ token rises: B_END → 1

const WORDS = ['Inference', 'is', 'one', 'token', 'at', 'a', 'time.'];

type Palette = { paper: string; paper2: string; ink: string; blue: string; amber: string; faint: string };
const readPalette = (): Palette => {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim() || '#000';
  return { paper: v('--paper'), paper2: v('--paper-2'), ink: v('--ink'), blue: v('--blue'), amber: v('--amber'), faint: v('--ink-faint') };
};

function usePalette() {
  const [p, setP] = useState<Palette>(() => readPalette());
  useEffect(() => {
    const on = () => setP(readPalette());
    window.addEventListener('themechange', on);
    document.addEventListener('astro:after-swap', on);
    return () => {
      window.removeEventListener('themechange', on);
      document.removeEventListener('astro:after-swap', on);
    };
  }, []);
  return p;
}

const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

const smPos = (i: number) => {
  const c = i % COLS;
  const r = Math.floor(i / COLS);
  return { c, x: -DIE_W / 2 + SM / 2 + c * (SM + GAP), z: -DIE_D / 2 + SM / 2 + r * (SM + GAP) };
};

// Lanes: from each HBM stack's inner edge to the die edge
const LANES = [-1, 1].flatMap((side) => STACK_Z.map((z) => ({ side, z, x0: side * (STACK_X - 0.6), x1: side * (DIE_W / 2 + 0.15) })));

/** Screen-space label anchors (world coordinates). Index 3 follows the token. */
const ANCHORS = [
  new THREE.Vector3(-STACK_X, 0.9, -1.85),
  new THREE.Vector3(STACK_X, 0.9, -1.85),
  new THREE.Vector3(0, 0.1, DIE_D / 2 + 0.5),
  new THREE.Vector3(),
];

function Scene({
  palette,
  animate,
  labels,
  onStep,
  onWord,
}: {
  palette: Palette;
  animate: boolean;
  labels: React.MutableRefObject<(HTMLElement | null)[]>;
  onStep: (s: number) => void;
  onWord: () => void;
}) {
  const clock = useRef(animate ? 0 : CYCLE * 0.86); // static frame: token mid-rise
  const stepRef = useRef(-1);
  const smRef = useRef<THREE.InstancedMesh>(null);
  const pkRef = useRef<THREE.InstancedMesh>(null);
  const tokenRef = useRef<THREE.Group>(null);
  const tmp = useMemo(() => new THREE.Object3D(), []);
  const cIdle = useMemo(() => new THREE.Color(), []);
  const cHot = useMemo(() => new THREE.Color(), []);
  const cMix = useMemo(() => new THREE.Color(), []);
  const { camera, pointer, size } = useThree();
  const proj = useMemo(() => new THREE.Vector3(), []);
  const placed = useRef(false);
  const camTarget = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    cIdle.set(palette.paper2);
    cHot.set(palette.blue);
  }, [palette, cIdle, cHot]);

  const outlines = useMemo(() => {
    const pts: number[] = [];
    for (let i = 0; i < COLS * ROWS; i++) {
      const { x, z } = smPos(i);
      const h = SM / 2;
      const y = 0.2;
      const k = [
        [x - h, y, z - h],
        [x + h, y, z - h],
        [x + h, y, z + h],
        [x - h, y, z + h],
      ];
      for (let j = 0; j < 4; j++) pts.push(...k[j], ...k[(j + 1) % 4]);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, []);
  const laneGeom = useMemo(() => {
    const pts: number[] = [];
    LANES.forEach((l) => pts.push(l.x0, 0.02, l.z, l.x1, 0.02, l.z));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    return g;
  }, []);
  const boxEdges = useMemo(
    () => ({
      board: new THREE.EdgesGeometry(new THREE.BoxGeometry(BOARD_W, 0.1, DIE_D + 1.6)),
      die: new THREE.EdgesGeometry(new THREE.BoxGeometry(DIE_W + 0.3, 0.04, DIE_D + 0.3)),
      layer: new THREE.EdgesGeometry(new THREE.BoxGeometry(1.1, 0.14, 1.3)),
    }),
    [],
  );

  useFrame((_, dt) => {
    if (animate) clock.current += Math.min(dt, 1 / 30);
    const t = clock.current;
    const u = (t % CYCLE) / CYCLE; // phase within one decode step
    const step = u < A_END ? 0 : u < B_END ? 1 : 2;
    if (step !== stepRef.current) {
      // a new cycle just began → the previous token has been emitted
      if (stepRef.current === 2 && step === 0) onWord();
      stepRef.current = step;
      onStep(step);
    }

    // ① weight packets glide along the lanes (conveyor, evenly spaced)
    const streamOn = smooth(0, 0.05, u) * (1 - smooth(A_END - 0.06, A_END, u));
    if (pkRef.current) {
      let i = 0;
      for (const lane of LANES) {
        for (let k = 0; k < PACKETS_PER_LANE; k++) {
          const s = (t * 0.55 + k / PACKETS_PER_LANE) % 1;
          const edge = smooth(0, 0.12, s) * (1 - smooth(0.88, 1, s));
          tmp.position.set(THREE.MathUtils.lerp(lane.x0, lane.x1, s), 0.09, lane.z);
          tmp.scale.setScalar(Math.max(0.0001, edge * streamOn));
          tmp.updateMatrix();
          pkRef.current.setMatrixAt(i++, tmp.matrix);
        }
      }
      pkRef.current.instanceMatrix.needsUpdate = true;
    }

    // ② a wave of work sweeps from both edges to the center, then the whole chip glows briefly
    const front = smooth(0.04, A_END, u); // 0 = edges, 1 = center
    const allBusy = smooth(A_END - 0.04, A_END + 0.03, u) * (1 - smooth(B_END, B_END + 0.12, u));
    if (smRef.current) {
      for (let i = 0; i < COLS * ROWS; i++) {
        const { c, x, z } = smPos(i);
        const d = Math.min(c, COLS - 1 - c) / ((COLS - 1) / 2); // 0 edge → 1 center
        const band = Math.max(0, 1 - Math.abs(d - front) * 3.2) * (u < A_END ? 1 : 0);
        const heat = Math.max(band, allBusy * 0.9);
        tmp.position.set(x, 0.1 + heat * 0.06, z);
        tmp.scale.set(1, 1 + heat * 0.6, 1);
        tmp.updateMatrix();
        smRef.current.setMatrixAt(i, tmp.matrix);
        cMix.copy(cIdle).lerp(cHot, heat * 0.85);
        smRef.current.setColorAt(i, cMix);
      }
      smRef.current.instanceMatrix.needsUpdate = true;
      if (smRef.current.instanceColor) smRef.current.instanceColor.needsUpdate = true;
    }

    // ③ the token rises out of the chip
    if (tokenRef.current) {
      const r = smooth(B_END - 0.04, 1, u);
      const vis = smooth(B_END - 0.04, B_END + 0.04, u) * (1 - smooth(0.97, 1, u));
      tokenRef.current.position.set(0, 0.45 + r * 1.6, 0);
      tokenRef.current.scale.setScalar(Math.max(0.0001, vis));
      tokenRef.current.visible = vis > 0.01;
      ANCHORS[3].set(0, tokenRef.current.position.y + 0.75, 0);
      const el = labels.current[3];
      if (el) el.style.opacity = String(vis);
    }

    // fit the whole board in view at a ~50° elevation, plus a gentle parallax
    const aspect = size.width / Math.max(1, size.height);
    const vHalf = THREE.MathUtils.degToRad(FOV / 2);
    const hHalf = Math.atan(Math.tan(vHalf) * aspect);
    const dist = Math.max(HALF_W / Math.tan(hHalf), HALF_D / Math.tan(vHalf)) + 2.2;
    camTarget.set(pointer.x * 0.6, Math.sin(ELEV) * dist + pointer.y * 0.3, Math.cos(ELEV) * dist);
    if (!placed.current) {
      camera.position.copy(camTarget); // no fly-in on load
      placed.current = true;
    } else camera.position.lerp(camTarget, 0.04);
    camera.lookAt(0, 0.4, 0);
    camera.updateMatrixWorld();

    // place the DOM labels over their 3D anchors
    ANCHORS.forEach((a, i) => {
      const el = labels.current[i];
      if (!el) return;
      proj.copy(a).project(camera);
      const x = ((proj.x + 1) / 2) * size.width;
      const y = ((1 - proj.y) / 2) * size.height;
      el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
    });
  });

  return (
    <group>
      {/* board */}
      <mesh position={[0, -0.07, 0]}>
        <boxGeometry args={[BOARD_W, 0.1, DIE_D + 1.6]} />
        <meshBasicMaterial color={palette.paper} />
      </mesh>
      <lineSegments position={[0, -0.07, 0]} geometry={boxEdges.board}>
        <lineBasicMaterial color={palette.ink} transparent opacity={0.5} />
      </lineSegments>
      <lineSegments position={[0, 0, 0]} geometry={boxEdges.die}>
        <lineBasicMaterial color={palette.faint} />
      </lineSegments>

      {/* lanes */}
      <lineSegments geometry={laneGeom}>
        <lineBasicMaterial color={palette.faint} transparent opacity={0.7} />
      </lineSegments>

      {/* SMs */}
      <instancedMesh ref={smRef} args={[undefined, undefined, COLS * ROWS]}>
        <boxGeometry args={[SM, 0.2, SM]} />
        <meshBasicMaterial color="#ffffff" />
      </instancedMesh>
      <lineSegments geometry={outlines}>
        <lineBasicMaterial color={palette.ink} transparent opacity={0.4} />
      </lineSegments>

      {/* HBM stacks */}
      {LANES.map((l, si) =>
        [0, 1, 2, 3].map((k) => (
          <group key={`${si}-${k}`} position={[l.side * STACK_X, 0.07 + k * 0.16, l.z]}>
            <mesh>
              <boxGeometry args={[1.1, 0.14, 1.3]} />
              <meshBasicMaterial color={k === 3 ? palette.paper2 : palette.paper} />
            </mesh>
            <lineSegments geometry={boxEdges.layer}>
              <lineBasicMaterial color={palette.ink} />
            </lineSegments>
          </group>
        )),
      )}

      {/* weight packets */}
      <instancedMesh ref={pkRef} args={[undefined, undefined, LANES.length * PACKETS_PER_LANE]}>
        <boxGeometry args={[0.2, 0.07, 0.22]} />
        <meshBasicMaterial color={palette.amber} />
      </instancedMesh>

      {/* the token */}
      <group ref={tokenRef}>
        <mesh>
          <boxGeometry args={[0.62, 0.62, 0.62]} />
          <meshBasicMaterial color={palette.blue} />
        </mesh>
      </group>

    </group>
  );
}

const LABEL =
  'absolute top-0 left-0 whitespace-nowrap rounded-full border px-2.5 py-1 font-mono text-[11px] leading-none tracking-wide shadow-sm will-change-transform';

const STEPS = [
  { n: '①', t: 'Weights stream in from memory', c: 'var(--amber)' },
  { n: '②', t: 'The cores do the math', c: 'var(--blue)' },
  { n: '③', t: 'One token comes out', c: 'var(--blue)' },
];

export default function GPUHero() {
  const palette = usePalette();
  const [reduced, setReduced] = useState(false);
  const [ready, setReady] = useState(false);
  const [step, setStep] = useState(0);
  const [count, setCount] = useState(0);
  const labels = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    const mq = matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);

  // count = tokens emitted so far. Keep the finished sentence visible while the next one starts.
  const k = count % WORDS.length;
  const shown = count > 0 && k === 0 ? WORDS : WORDS.slice(0, k);

  return (
    <div className="flex h-full w-full flex-col" style={{ opacity: ready ? 1 : 0, transition: 'opacity 1s ease' }}>
      <div className="relative min-h-0 flex-1">
        <Canvas
          dpr={[1, 2]}
          camera={{ position: [0, 10, 8], fov: FOV }}
          gl={{ antialias: true, alpha: true }}
          onCreated={() => {
            setReady(true);
            document.getElementById('hero-sketch')?.style.setProperty('opacity', '0');
          }}
          role="img"
          aria-label="Animation of one decode step: weights stream from HBM memory into the GPU's compute cores, the cores light up, and one token comes out."
        >
          <Scene palette={palette} animate={!reduced} labels={labels} onStep={setStep} onWord={() => setCount((n) => n + 1)} />
        </Canvas>
        {/* labels: plain DOM, positioned each frame by <Scene> */}
        <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
          {[
            { text: 'HBM · memory', border: 'var(--amber)' },
            { text: 'HBM · memory', border: 'var(--amber)' },
            { text: 'SMs · compute cores', border: 'var(--blue)' },
          ].map((l, i) => (
            <span key={i} ref={(el) => void (labels.current[i] = el)} className={`${LABEL} bg-paper text-ink`} style={{ borderColor: l.border }}>
              {l.text}
            </span>
          ))}
          <span
            ref={(el) => void (labels.current[3] = el)}
            className={`${LABEL} !px-3 !py-1.5 !text-[13px]`}
            style={{ background: 'var(--blue)', color: 'var(--paper)', borderColor: 'var(--blue)', opacity: 0 }}
          >
            “{WORDS[count % WORDS.length]}”
          </span>
        </div>
      </div>

      {/* synced legend */}
      <div className="px-2 pb-1">
        <ol className="grid grid-cols-3 gap-2">
          {STEPS.map((s, i) => {
            const on = i === step;
            return (
              <li
                key={s.n}
                className="rounded-md border border-dashed px-2.5 py-1.5 text-[0.78rem] leading-snug transition-all duration-500"
                style={{
                  borderColor: on ? s.c : 'var(--rule)',
                  background: on ? `color-mix(in srgb, ${s.c} 10%, transparent)` : 'transparent',
                  opacity: on ? 1 : 0.55,
                }}
              >
                <span className="mr-1 font-mono" style={{ color: s.c }}>
                  {s.n}
                </span>
                {s.t}
              </li>
            );
          })}
        </ol>
        <p className="mt-2 flex min-h-[1.6rem] items-baseline gap-2 font-mono text-[0.72rem] text-ink-faint">
          <span className="shrink-0">output so far:</span>
          <span className="font-serif text-[0.95rem] text-ink">
            {shown.join(' ')}
            <span className="ml-0.5 inline-block h-3.5 w-[2px] translate-y-0.5 animate-pulse bg-ink" />
          </span>
        </p>
      </div>
    </div>
  );
}
