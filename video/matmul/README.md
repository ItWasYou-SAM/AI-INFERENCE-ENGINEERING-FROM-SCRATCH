# Matmul: 57× on a laptop

A 22.5-second, 1080p narrated cut about one demanding topic: chapters [6.1](../../src/content/chapters/m06/tiled-matmul.mdx)
and [6.2](../../src/content/chapters/m06/register-tiling.mdx), the same 4096 × 4096 FP32 matmul written five ways.
It uses the same look, synthesizer and narrator as the [promo](../promo/README.md). Every number on screen is measured output
printed in those chapters (RTX 3050 6GB Laptop).

| Time | Scene | Narrator |
|:--|:--|:--|
| 0:00 | Naive kernel: one output at a time, 1,821 ms | "On a laptop GPU, a naive matrix multiply takes two seconds." |
| 0:04.4 | Coalescing: 32 scattered sectors → one 128-byte load, 75 → 535 GFLOP/s | "Change one index, and the memory reads coalesce. Seven times faster." |
| 0:08.8 | Tiles fly into shared memory, then 8 × 8 register tiles, → 2,973 | "Stage tiles in shared memory, then registers. Reuse every load." |
| 0:13.4 | All five kernels against cuBLAS: 57×, 94% of cuBLAS | "Fifty-seven times faster. Ninety-four percent of NVIDIA's own library." |
| 0:18.0 | Call to action: GitHub link | "Write every kernel yourself. Free and open source, on GitHub." |

Scene times (`TL`), narration (`NARRATION`) and the measured numbers (`K`) are at the top of the script in `index.html`.

## Rebuild

Same tools as the promo (Chromium, ffmpeg, NumPy, `puppeteer-core` in `PPTR_HOME`, Kokoro for the voice).
The promo's `render.mjs` and `narrate.py` render this folder when `VIDEO=matmul` is set:

```bash
export VIDEO=matmul
node video/promo/render.mjs --meta                              # out/events.json + out/timeline.json
~/.cache/kokoro-tts/venv/bin/python video/promo/narrate.py      # out/narration/*.wav
python3 video/matmul/soundtrack.py                              # out/soundtrack.wav
node video/promo/render.mjs video/matmul/out/silent.mp4         # frames → video
ffmpeg -i video/matmul/out/silent.mp4 -i video/matmul/out/soundtrack.wav \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart video/matmul/out/matmul.mp4
ffmpeg -i video/matmul/out/matmul.mp4 -c:v libx264 -preset slow -crf 26 -c:a aac -b:a 160k \
  -movflags +faststart video/matmul/out/matmul-web.mp4             # ~5 MB, for X / LinkedIn / README
node video/promo/render.mjs --stills 2,6,12                     # quick PNG stills
```

Preview in a browser by serving `video/` and opening `/matmul/index.html` (`?t=10` jumps to a second).
