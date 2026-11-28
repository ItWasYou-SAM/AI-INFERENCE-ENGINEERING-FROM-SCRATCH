// Capture chapter 6.1 from the live site, dark theme, 1280x720 CSS px at 2x (2560x1440 images).
//
//   PPTR_HOME=... node video/chapter-6-1/capture.mjs     -> shots/*.jpg + shots/shots.json
//
// Tall shots (title, naive, measured) are page slices with the sticky header hidden, plus the header
// on its own, so the edit can scroll smoothly under a fixed header. The figure is captured once per
// real state, clicking its real buttons; shots.json records where each button sits on screen.
import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.PPTR_HOME || process.cwd(), 'noop.js'));
const puppeteer = require('puppeteer-core');
const SITE = process.env.SITE || 'https://ai-inference-engineering-from-scrat.vercel.app';
const CH = SITE + '/learn/m06/tiled-matmul';
const VW = 1280, VH = 720, DPR = 2;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const out = (n) => path.join(here, 'shots', n + '.jpg');
const meta = { viewport: [VW, VH], dpr: DPR, site: SITE };

const b = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/chromium', headless: true, args: ['--hide-scrollbars'] });
const p = await b.newPage();
await p.setViewport({ width: VW, height: VH, deviceScaleFactor: DPR });
await p.evaluateOnNewDocument(() => localStorage.setItem('theme', 'dark'));

async function open(url) {
  await p.goto(url, { waitUntil: 'networkidle2' });
  await p.evaluate(() => document.fonts.ready);
  // reveal every scroll-triggered block, as a reader would see it after scrolling
  await p.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 400) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 60)); }
    window.scrollTo(0, 0);
  });
  await sleep(1200);
}
const topOf = (pred) => p.evaluate((src) => {
  const f = new Function('el', 'return ' + src);
  const el = [...document.querySelectorAll('h1,h2,button,pre,figure')].find(f);
  return el.getBoundingClientRect().top + scrollY;
}, pred);
const headerH = () => p.evaluate(() => document.querySelector('header').getBoundingClientRect().height);
async function slice(name, y0, h) {
  await p.evaluate(() => { document.querySelector('header').style.visibility = 'hidden'; });
  await p.screenshot({ path: out(name), type: 'jpeg', quality: 92, clip: { x: 0, y: y0, width: VW, height: h }, captureBeyondViewport: true });
  await p.evaluate(() => { document.querySelector('header').style.visibility = ''; });
  meta[name] = { y0, h };
}
async function view(name, extra = {}) {
  await sleep(350);
  await p.screenshot({ path: out(name), type: 'jpeg', quality: 92 });
  meta[name] = extra;
}
const btn = (text) => p.evaluate((t) => {
  const el = [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === t);
  const r = el.getBoundingClientRect(); return [Math.round(r.x + r.width / 2), Math.round(r.y + r.height / 2)];
}, text);
const click = async (text) => { const xy = await btn(text); await p.mouse.click(xy[0], xy[1]); await sleep(400); return xy; };

/* chapter page */
await open(CH);
await p.evaluate(() => window.scrollTo(0, 0)); await sleep(500);
await p.screenshot({ path: out('header'), type: 'jpeg', quality: 92, clip: { x: 0, y: 0, width: VW, height: Math.ceil(await headerH()) } });
meta.header = { h: Math.ceil(await headerH()) };

await slice('title', 0, 1500);                                         // title, summary, objectives
const v1 = await topOf("el.tagName==='H2' && el.textContent.startsWith('Version 1')");
await slice('naive', v1 - 80, 1150);                                   // naive kernel code -> coalesced
const ms = await topOf("el.tagName==='H2' && el.textContent.startsWith('Measured')");
await slice('measured', ms - 80, 900);                                 // measured output table

/* where the highlighted phrases sit, in page px: a phrase in prose, or a whole line of a code block */
meta.marks = await p.evaluate(() => {
  const page = (r) => [r.x, r.y + scrollY, r.width, r.height].map(Math.round);
  const phrase = (needle) => {
    const w = document.createTreeWalker(document.querySelector('main') || document.body, NodeFilter.SHOW_TEXT);
    for (let n; (n = w.nextNode());) {
      const i = n.data.indexOf(needle);
      if (i >= 0) { const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + needle.length); return page(r.getClientRects()[0]); }
    }
    return null;
  };
  const codeLine = (needle) => {
    const el = [...document.querySelectorAll('pre .ec-line, pre code > *')].find((l) => l.textContent.includes(needle));
    return el ? page(el.getBoundingClientRect()) : null;
  };
  return {
    gap: phrase('is 60×'), loop: codeLine('for (int k = 0'), sectors: phrase('32 separate sectors for 128 useful bytes'),
    naive: codeLine('naive        1821'), cublas: codeLine('cuBLAS         30'),
  };
});
for (const [k, v] of Object.entries(meta.marks)) if (!v) throw new Error('phrase not found: ' + k);

/* the interactive figure, every state, clicked for real */
const fb = await topOf("el.tagName==='BUTTON' && el.textContent.includes('one output per thread')");
await p.evaluate((y) => window.scrollTo(0, y), fb - 150); await sleep(900);
await view('fig-0');                                                   // default: tiled, 4x4, k-step 1
const B = {};
B.untiled = await click('one output per thread, no reuse'); await view('fig-untiled');
B.tiled = await click('shared-memory tiles'); await view('fig-tiled');
B.next = await btn('next k →');
for (let k = 2; k <= 4; k++) { await click('next k →'); await view(`fig-k${k}`); }
B.t8 = await click('8×8'); await view('fig-8-k1');
await click('next k →'); await view('fig-8-k2');                       // 8x8 has 2 k-steps
meta.buttons = B;

meta.star = await p.evaluate(() => {
  const a = [...document.querySelectorAll('header a')].find((a) => a.href.includes('github.com'));
  const r = a.getBoundingClientRect(); return [Math.round(r.x + r.width - 26), Math.round(r.y + r.height / 2)];
});

writeFileSync(path.join(here, 'shots', 'shots.json'), JSON.stringify(meta, null, 2));
console.log(JSON.stringify(meta));
await b.close();
