# Trailer

An 80-second, 1080p trailer for the course, drawn on a canvas and rendered frame by frame.

| Time | Scene | Idea |
|:--|:--|:--|
| 0:00 | Cold open | "Why does a chatbot write one word at a time?" |
| 0:08 | Title | Inference Engineering, from Scratch |
| 0:14 | 01 Tokens | Text → token IDs → next token |
| 0:22 | 02 The memory wall | Each token reads all 14 GB of weights: decode is memory-bound |
| 0:32 | 03 Prefill & decode | TTFT and TPOT |
| 0:40 | 04 The KV cache | Cache instead of recompute; the 24 GB budget; FP8 doubles users |
| 0:50 | 05 Continuous batching | Static vs continuous, GPU busy % |
| 0:59 | 06 The roofline | Decode on the slope, prefill on the roof, batching climbs |
| 1:06 | The course | 4 levels · 18 modules · 109 chapters |
| 1:13 | Call to action | GitHub link, start with Chapter 0.1 |

## Preview

Open `index.html` through any local server (for example `npx serve video/trailer`). It plays in real
time; click to restart, or add `?t=40` to jump to a second.

## Render

Needs Chromium, ffmpeg, Python 3 with NumPy, and `puppeteer-core` installed somewhere
(it is not a site dependency).

```bash
mkdir -p /tmp/pptr && (cd /tmp/pptr && npm i puppeteer-core)
export PPTR_HOME=/tmp/pptr

node video/trailer/render.mjs video/trailer/out/silent.mp4   # frames → video, also writes out/events.json
python3 video/trailer/soundtrack.py                           # events → out/soundtrack.wav
ffmpeg -i video/trailer/out/silent.mp4 -i video/trailer/out/soundtrack.wav \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart video/trailer/out/trailer.mp4

node video/trailer/render.mjs --stills 20,45,62               # quick PNG stills
```

All numbers on screen are worked examples: a 7B model in FP16 (14 GB), a 24 GB GPU, ~1 TB/s
memory bandwidth, and a Llama-2-7B-shaped KV cache (32 layers × 4096 dims, 4K tokens ≈ 2 GB per user).
