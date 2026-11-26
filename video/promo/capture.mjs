// Capture real screenshots of the running site for the promo video and images.
//
//   npm run build && npx astro preview --port 4399      (in another terminal)
//   PPTR_HOME=/path/with/node_modules BASE=http://localhost:4399 node video/promo/capture.mjs
//
// Writes JPEGs into video/promo/shots/. Every image is the real site, driven by
// clicks and slider moves, at 1440 x 900 and 2x pixel density.
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.PPTR_HOME || process.cwd(), 'noop.js'));
const puppeteer = require('puppeteer-core');
const BASE = process.env.BASE || 'http://localhost:4399';
const OUT = path.join(here, 'shots');
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || '/usr/bin/chromium',
  headless: true,
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--hide-scrollbars'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
page.on('pageerror', (e) => console.error('page error:', e.message));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(url, theme = 'light') {
  await page.evaluateOnNewDocument((th) => localStorage.setItem('theme', th), theme);
  await page.goto(BASE + url, { waitUntil: 'networkidle0' });
  await page.evaluate(() => document.fonts.ready);
  // reveal scroll-triggered content and settle
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 60)); }
    window.scrollTo(0, 0);
  });
  await sleep(700);
}
const shot = async (name, opts = {}) => {
  await page.screenshot({ path: path.join(OUT, name + '.jpg'), type: 'jpeg', quality: 90, ...opts });
  console.log(name);
};
// full-page shot: make the viewport as tall as the page so every scroll-reveal animation fires
async function fullShot(name) {
  const h = await page.evaluate(() => document.documentElement.scrollHeight);
  await page.setViewport({ width: 1440, height: h, deviceScaleFactor: 2 });
  await sleep(2500);
  await shot(name, { fullPage: true });
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
  await sleep(300);
}
async function fig(n = 0) {
  await page.addStyleTag({ content: 'header, astro-dev-toolbar { display: none !important; }' });
  const figs = await page.$$('figure');
  const el = figs[n];
  await el.evaluate((e) => e.scrollIntoView({ block: 'center' }));
  await sleep(500);
  return el;
}
const figShot = async (el, name) => {
  await sleep(450);
  await el.screenshot({ path: path.join(OUT, name + '.jpg'), type: 'jpeg', quality: 92 });
  console.log(name);
};
async function clickText(scope, text) {
  const ok = await scope.evaluate((root, text) => {
    const b = [...root.querySelectorAll('button')].find((x) => x.textContent.trim().includes(text));
    if (b) b.click();
    return !!b;
  }, text);
  if (!ok) console.warn('no button', text);
}
async function setRange(scope, idx, value) {
  await scope.evaluate((root, idx, value) => {
    const r = root.querySelectorAll('input[type=range]')[idx];
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(r, String(value));
    r.dispatchEvent(new Event('input', { bubbles: true }));
  }, idx, value);
}

// ── pages ──────────────────────────────────────────────────────────────
await open('/');
await sleep(2500);                                        // let the 3D GPU spin up
await shot('home-hero');
await fullShot('home-full');
await open('/', 'dark');
await sleep(2500);
await shot('home-hero-dark');

await open('/roadmap');
await shot('roadmap');
await fullShot('roadmap-full');

await open('/learn/m05/softmax-kernels');
await shot('chapter-top');
await open('/learn/m05/softmax-kernels', 'dark');
await shot('chapter-top-dark');

// ── interactive figures ────────────────────────────────────────────────
await open('/learn/m00/life-of-a-request');
let el = await fig(0);
// step through the stages: request, tokenize, schedule, prefill, then decode tokens one at a time
const stepsAt = [1, 2, 3, 4, 7, 11];
for (let k = 1, i = 0; k <= 11; k++) {
  await clickText(el, 'Step');
  if (k === stepsAt[i]) await figShot(el, `fig-request-${i++}`);
}

await open('/learn/m09/kv-cache-size');
el = await fig(0);
for (const [i, u] of [[0, 8], [1, 40], [2, 80], [3, 107], [4, 150]]) {   // A100-80GB fits 107 at 4K
  await setRange(el, 1, u);
  await figShot(el, `fig-kv-${i}`);
}
await el.evaluate((root) => {                              // the KV-cache FP8 button (the last "FP8")
  const b = [...root.querySelectorAll('button')].filter((x) => x.textContent.trim() === 'FP8').pop();
  b.click();
});
await figShot(el, 'fig-kv-5');

await open('/learn/m03/roofline');
el = await fig(0);
for (const [i, b] of [[0, 0], [1, 3], [2, 6], [3, 9]]) {
  await setRange(el, 0, b);
  await figShot(el, `fig-roofline-${i}`);
}

await open('/learn/m01/attention-step-by-step');
el = await fig(0);
await figShot(el, 'fig-attention-0');
await clickText(el, 'next →'); await clickText(el, 'next →');
await figShot(el, 'fig-attention-1');

await open('/learn/m05/reduction-trees');
el = await fig(0);
for (let i = 0; i < 5; i++) { await figShot(el, `fig-reduce-${i}`); await clickText(el, 'next →'); }
await clickText(el, 'v3 sequential');
await clickText(el, 'next →'); await clickText(el, 'next →');
await figShot(el, 'fig-reduce-v3');

await open('/learn/m05/prefix-sum-scan');
el = await fig(0);
await clickText(el, 'Blelloch');
for (let i = 0; i <= 9; i++) { if (i === 0 || i === 3 || i === 5 || i === 9) await figShot(el, `fig-scan-${i}`); await clickText(el, 'next →'); }

await open('/learn/m04/coalesced-vs-strided');
el = await fig(0);
await figShot(el, 'fig-coalesce-0');
await clickText(el, 'x[8 * i]');
await figShot(el, 'fig-coalesce-1');

// ── a quiz, answered ───────────────────────────────────────────────────
await open('/learn/m05/softmax-kernels');
await page.addStyleTag({ content: 'header, astro-dev-toolbar { display: none !important; }' });
const quiz = await page.evaluateHandle(() => document.querySelector('astro-island[component-url*="Quiz"]').firstElementChild);
await quiz.evaluate((q) => q.scrollIntoView({ block: 'center' }));
await sleep(600);
await quiz.screenshot({ path: path.join(OUT, 'quiz-0.jpg'), type: 'jpeg', quality: 92 });
await page.evaluate(() => {
  const q = document.querySelector('astro-island[component-url*="Quiz"]');   // first question, correct answer
  const btn = [...q.querySelectorAll('button')].find((b) => b.textContent.includes('prevents exp()'));
  btn?.click();
});
await sleep(900);
await quiz.screenshot({ path: path.join(OUT, 'quiz-1.jpg'), type: 'jpeg', quality: 92 });
console.log('quiz');

await browser.close();
