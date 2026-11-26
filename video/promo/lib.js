// Shared drawing helpers for the promo video (index.html) and the promo images (images.html).
// Same palette, fonts and grain as the trailer, so the two read as one family.

const C = {
  paper: '#121720', paper2: '#19202b', paper3: '#222b38', ink: '#ebe5d6', soft: '#a9a293',
  faint: '#767063', rule: '#2d3644', blue: '#84adff', red: '#ff8a78', green: '#73d39b',
  amber: '#f0c06a', violet: '#b6a0ff', teal: '#6fd8d0', site: '#f4efe4',
};
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const seg = (t, a, b) => clamp((t - a) / (b - a));
const lerp = (a, b, p) => a + (b - a) * p;
const eo = (x) => 1 - Math.pow(1 - x, 3);
const eio = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}
function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

let ctx, W, H;
function useCanvas(cv) { ctx = cv.getContext('2d'); W = cv.width; H = cv.height; }

const FAM = { d: 'Fraunces', m: 'JBM', s: 'Serif4' };
const font = (size, weight = 500, fam = 'd', italic = false) => `${italic ? 'italic ' : ''}${weight} ${size}px ${FAM[fam]}`;
function measure(str, f) { ctx.save(); ctx.font = f; const w = ctx.measureText(str).width; ctx.restore(); return w; }
function T(str, x, y, o = {}) {
  ctx.save();
  ctx.globalAlpha *= o.a ?? 1;
  ctx.font = o.f || font(32);
  ctx.fillStyle = o.c || C.ink;
  ctx.textAlign = o.align || 'left';
  ctx.textBaseline = o.base || 'alphabetic';
  if (o.ls) ctx.letterSpacing = o.ls + 'px';
  if (o.glow) { ctx.shadowColor = o.glow; ctx.shadowBlur = o.blur || 24; }
  ctx.fillText(str, x, y);
  ctx.restore();
}
// word-wrapped text; returns the y below the last line
function wrap(str, x, y, maxW, lineH, o = {}) {
  const f = o.f || font(32);
  const words = str.split(' ');
  let lineStr = '';
  const lines = [];
  for (const w of words) {
    const test = lineStr ? lineStr + ' ' + w : w;
    if (measure(test, f) > maxW && lineStr) { lines.push(lineStr); lineStr = w; } else lineStr = test;
  }
  if (lineStr) lines.push(lineStr);
  lines.forEach((l, i) => T(l, x, y + i * lineH, { ...o, f }));
  return y + lines.length * lineH;
}
function rr(x, y, w, h, r, fill, stroke, lw = 2) {
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}
function line(x1, y1, x2, y2, col, lw = 2) {
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = lw;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
}

// ── images ─────────────────────────────────────────────────────────────
const SHOT_NAMES = [
  'home-hero', 'home-hero-dark', 'home-full', 'roadmap', 'roadmap-full', 'chapter-top', 'chapter-top-dark',
  'quiz-0', 'quiz-1', 'fig-attention-0', 'fig-attention-1', 'fig-coalesce-0', 'fig-coalesce-1',
  ...[0, 1, 2, 3, 4, 5].map((i) => `fig-request-${i}`), ...[0, 1, 2, 3, 4, 5].map((i) => `fig-kv-${i}`),
  ...[0, 1, 2, 3].map((i) => `fig-roofline-${i}`), ...[0, 1, 2, 3, 4].map((i) => `fig-reduce-${i}`), 'fig-reduce-v3',
  ...[0, 3, 5, 9].map((i) => `fig-scan-${i}`),
];
const IMG = {};
for (const n of SHOT_NAMES) { const im = new Image(); im.src = `shots/${n}.jpg`; IMG[n] = im; }
const logo = new Image(); logo.src = '../trailer/logo-128.png';
const imagesReady = Promise.all([...Object.values(IMG), logo].map((im) => new Promise((r) => { if (im.complete) r(); else { im.onload = r; im.onerror = r; } })));
const fontsReady = Promise.all([
  document.fonts.load(font(40, 500)), document.fonts.load(font(40, 500, 'd', true)),
  document.fonts.load(font(40, 500, 'm')), document.fonts.load(font(40, 400, 's')),
]);

// grain tiles
const grain = [0, 1, 2, 3].map((k) => {
  const g = document.createElement('canvas'); g.width = g.height = 256;
  const gx = g.getContext('2d'); const im = gx.createImageData(256, 256); const r = mulberry(99 + k);
  for (let i = 0; i < im.data.length; i += 4) { const v = r() * 255; im.data[i] = im.data[i + 1] = im.data[i + 2] = v; im.data[i + 3] = 255; }
  gx.putImageData(im, 0, 0); return g;
});

// ── shared pieces ──────────────────────────────────────────────────────
function background(t = 0, lift = 1) {
  ctx.fillStyle = '#0d1118'; ctx.fillRect(0, 0, W, H);
  const s = Math.max(W, H) / 1920;
  const blobs = [
    [C.teal, W * 0.27 + 140 * s * Math.sin(t * 0.11), H * 0.35 + 80 * s * Math.cos(t * 0.09), 620 * s],
    [C.violet, W * 0.74 + 120 * s * Math.cos(t * 0.08), H * 0.67 + 90 * s * Math.sin(t * 0.12), 680 * s],
    [C.amber, W * 0.5 + 300 * s * Math.sin(t * 0.05), H * 0.91, 520 * s],
  ];
  for (const [c, x, y, r] of blobs) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, rgba(c, 0.1 * lift)); g.addColorStop(1, rgba(c, 0));
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  }
  ctx.save(); ctx.globalAlpha = lift;
  const off = (t * 6) % 48;
  ctx.strokeStyle = 'rgba(122,167,255,0.045)'; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = -48 + off; x < W + 48; x += 48) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  for (let y = -48 + off * 0.5; y < H + 48; y += 48) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  ctx.stroke(); ctx.restore();
}
function finishFrame(t = 0, vignette = 0.62) {
  const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.62);
  g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, `rgba(0,0,0,${vignette})`);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  ctx.save(); ctx.globalAlpha = 0.05; ctx.globalCompositeOperation = 'screen';
  ctx.fillStyle = ctx.createPattern(grain[Math.floor(t * 24) % 4], 'repeat'); ctx.fillRect(0, 0, W, H); ctx.restore();
}

// A browser window with a screenshot inside. src: { img, sx, sy, sw } (crop in image pixels; sh follows the box).
function browser(x, y, w, h, url, draw, o = {}) {
  const bar = o.bar ?? 46, r = o.r ?? 16;
  ctx.save();
  ctx.globalAlpha *= o.a ?? 1;
  ctx.shadowColor = 'rgba(0,0,0,0.55)'; ctx.shadowBlur = 60; ctx.shadowOffsetY = 24;
  rr(x, y, w, h, r, o.dark ? '#121720' : C.site);
  ctx.restore();
  ctx.save();
  ctx.globalAlpha *= o.a ?? 1;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.clip();
  // title bar
  ctx.fillStyle = '#1b212b'; ctx.fillRect(x, y, w, bar);
  ['#ff8a78', '#f0c06a', '#73d39b'].forEach((c, i) => { ctx.beginPath(); ctx.arc(x + 26 + i * 22, y + bar / 2, 6.5, 0, 7); ctx.fillStyle = c; ctx.fill(); });
  if (url) {
    const uw = Math.min(w * 0.55, 620);
    rr(x + (w - uw) / 2, y + 9, uw, bar - 18, (bar - 18) / 2, '#0f141b');
    T(url, x + w / 2, y + bar / 2 + 6, { f: font(16, 400, 'm'), c: C.soft, align: 'center' });
  }
  ctx.beginPath(); ctx.rect(x, y + bar, w, h - bar); ctx.clip();
  draw(x, y + bar, w, h - bar);
  ctx.restore();
  ctx.save(); ctx.globalAlpha *= o.a ?? 1; rr(x + 0.5, y + 0.5, w - 1, h - 1, r, null, 'rgba(255,255,255,0.08)', 1); ctx.restore();
}
// draw image crop (sx, sy, sw) scaled to width w at (x, y)
function crop(img, x, y, w, h, sx = 0, sy = 0, sw = null, a = 1) {
  if (!img || !img.naturalWidth) return;
  sw = sw ?? img.naturalWidth;
  const k = w / sw;
  const sh = Math.min(h / k, img.naturalHeight - sy);
  ctx.save(); ctx.globalAlpha *= a;
  ctx.drawImage(img, sx, sy, sw, sh, x, y, w, sh * k);
  ctx.restore();
}
// A paper card holding a figure screenshot, scaled to fit (w, h), top-aligned.
function figCard(x, y, w, h, img, a = 1) {
  ctx.save(); ctx.globalAlpha *= a;
  ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 50; ctx.shadowOffsetY = 20;
  rr(x, y, w, h, 18, C.site);
  ctx.restore();
  if (img?.naturalWidth) {
    const k = Math.min((w - 36) / img.naturalWidth, (h - 36) / img.naturalHeight);
    const iw = img.naturalWidth * k, ih = img.naturalHeight * k;
    ctx.save(); ctx.globalAlpha *= a;
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 18); ctx.clip();
    ctx.drawImage(img, x + (w - iw) / 2, y + 18, iw, ih);
    ctx.restore();
    return { x: x + (w - iw) / 2, y: y + 18, w: iw, h: ih };
  }
  return { x, y, w, h };
}
function cursor(x, y, press = 0, a = 1) {
  ctx.save(); ctx.globalAlpha *= a;
  if (press > 0) {
    ctx.beginPath(); ctx.arc(x, y, 14 + 30 * press, 0, 7);
    ctx.strokeStyle = rgba(C.amber, 0.9 * (1 - press)); ctx.lineWidth = 3; ctx.stroke();
  }
  ctx.translate(x, y);
  const s = 1 - 0.12 * Math.sin(Math.PI * clamp(press * 2.5));
  ctx.scale(s * 1.35, s * 1.35);
  ctx.beginPath();
  ctx.moveTo(0, 0); ctx.lineTo(0, 22); ctx.lineTo(5.5, 17); ctx.lineTo(9.5, 26); ctx.lineTo(13, 24.5); ctx.lineTo(9, 16); ctx.lineTo(16, 16); ctx.closePath();
  ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 8; ctx.shadowOffsetY = 3;
  ctx.fillStyle = '#fff'; ctx.fill();
  ctx.shadowColor = 'transparent'; ctx.lineWidth = 1.5; ctx.strokeStyle = '#111'; ctx.stroke();
  ctx.restore();
}
function pill(str, x, y, col, o = {}) {
  const f = o.f || font(22, 500, 'm');
  const w = measure(str, f) + 44, h = o.h || 48;
  const x0 = o.align === 'center' ? x - w / 2 : x;
  ctx.save(); ctx.globalAlpha *= o.a ?? 1;
  rr(x0, y, w, h, h / 2, rgba(col, 0.12), rgba(col, 0.8), 2);
  T(str, x0 + w / 2, y + h / 2 + 8, { f, c: o.c || C.ink, align: 'center' });
  ctx.restore();
  return w;
}
function titleLockup(cx, y, size, a = 1) {
  const tf1 = font(size, 560), tf2 = font(size, 480, 'd', true);
  const w1 = measure('Inference Engineering, ', tf1), w2 = measure('from Scratch', tf2);
  const tx = cx - (w1 + w2) / 2;
  T('Inference Engineering, ', tx, y, { f: tf1, a });
  T('from Scratch', tx + w1, y, { f: tf2, c: C.amber, a, glow: rgba(C.amber, 0.35), blur: 36 });
}
const URL_GH = 'github.com/ItWasYou-SAM/AI-INFERENCE-ENGINEERING-FROM-SCRATCH';

const USE_CASES = [
  ['01', 'Explain why decode is memory-bound', 'and do the math on paper, the way ML-systems interviews ask.'],
  ['02', 'Write GPU kernels that run fast', 'reductions, scans, softmax, matmul and attention, in CUDA.'],
  ['03', 'See how a chatbot is served', 'KV caches, batching and routing, one interactive figure at a time.'],
  ['04', 'Build and tune a serving engine', 'paged KV cache, continuous batching, speculative decoding.'],
  ['05', 'Cut the cost of every token', 'quantization, capacity planning, cost per million tokens.'],
  ['06', 'Start without a GPU', 'all of Level 1 runs on a laptop; GPU labs run on free Colab.'],
];
