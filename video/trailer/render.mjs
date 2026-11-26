// Render index.html frame by frame into an MP4 (plus events.json for the soundtrack).
//
//   npm i puppeteer-core              (anywhere; point PPTR_HOME at that folder)
//   PPTR_HOME=/path/with/node_modules node video/trailer/render.mjs [out.mp4]
//   ... render.mjs --stills 3,10,25   writes PNG stills instead, for a quick look
//
// Needs Chromium (CHROME=/usr/bin/chromium by default) and ffmpeg on PATH.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { writeFileSync, existsSync, readFileSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.PPTR_HOME || process.cwd(), 'noop.js'));
const puppeteer = require('puppeteer-core');

const FPS = +(process.env.FPS || 30);
const args = process.argv.slice(2);
const stillsIdx = args.indexOf('--stills');
const stills = stillsIdx >= 0 ? args[stillsIdx + 1].split(',').map(Number) : null;
const out = args.find(a => a.endsWith('.mp4')) || path.join(here, 'out', 'silent.mp4');

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME || '/usr/bin/chromium',
  headless: true,
  args: ['--force-device-scale-factor=1', '--disable-gpu-vsync'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
page.on('pageerror', e => { console.error('page error:', e.message); process.exitCode = 1; });
// serve this folder over HTTP: images loaded from file:// would taint the canvas
const MIME = { '.html': 'text/html', '.png': 'image/png', '.woff2': 'font/woff2' };
const server = http.createServer((req, res) => {
  const f = path.join(here, path.normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^([.][.][\/\\])+/, ''));
  if (!f.startsWith(here) || !existsSync(f)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': MIME[path.extname(f)] || 'application/octet-stream' }).end(readFileSync(f));
}).listen(0, '127.0.0.1');
await new Promise(r => server.once('listening', r));
await page.goto(`http://127.0.0.1:${server.address().port}/index.html?render=1`);
await page.evaluate(() => window.ready);

const grab = (t, type, q) => page.evaluate((t, type, q) => {
  window.draw(t);
  return document.getElementById('cv').toDataURL(type, q).split(',')[1];
}, t, type, q);

if (stills) {
  for (const t of stills) {
    const f = path.join(here, 'out', `still-${t.toFixed(1).padStart(4, '0')}.png`);
    writeFileSync(f, Buffer.from(await grab(t, 'image/png'), 'base64'));
    console.log(f);
  }
  await browser.close(); server.close();
  process.exit();
}

const D = await page.evaluate(() => window.DURATION);
writeFileSync(path.join(here, 'out', 'events.json'), JSON.stringify(await page.evaluate(() => window.EVENTS)));

const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
  '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
const N = Math.round(D * FPS);
const t0 = Date.now();
for (let i = 0; i < N; i++) {
  const buf = Buffer.from(await grab(i / FPS, 'image/jpeg', 0.95), 'base64');
  if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  if (i % 150 === 0) console.log(`frame ${i}/${N}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
ff.stdin.end();
await new Promise(r => ff.on('close', r));
await browser.close(); server.close();
console.log('wrote', out, existsSync(out));
