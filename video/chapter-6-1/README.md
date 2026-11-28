# Chapter 6.1, live

A 22.5-second, 1080p narrated walk-through of [chapter 6.1](../../src/content/chapters/m06/tiled-matmul.mdx), cut from
**real captures of the live site** (dark theme, 1280 × 720 at 2×). Nothing on the page is redrawn: the edit adds camera moves,
a cursor, highlighter swipes, labels in the letterbox, narration and music.

| Time | Shot (all from the live site) | Narrator |
|:--|:--|:--|
| 0:00 | Chapter title and summary; "is 60×" highlighted | "A naive matrix multiply is sixty times slower than NVIDIA's best." |
| 0:04.4 | The naive kernel's code and why it is slow | "Chapter 6.1 closes the gap, starting with memory access." |
| 0:08.6 | The tiling figure, its real buttons clicked: loads 128 → 32, then k-steps and 8 × 8 | "Then you try it yourself: tile the matrix, and watch the memory loads drop." |
| 0:14.6 | The measured output: naive 1821 ms, cuBLAS 30.55 ms | "Every number is measured, on a laptop GPU." |
| 0:18.4 | The cursor stars the repo from the site header; end card | "Free and open source. Star it on GitHub." |

## How it is made

`capture.mjs` opens the live chapter in Chromium and saves `shots/`: tall page slices (header hidden, so the edit can scroll),
one screenshot per state of the figure after clicking its real buttons, and `shots.json` with each button's position and
the position of every highlighted phrase, read from the DOM. Headless Chrome's own screencast is limited to CSS resolution
and ~16 fps, so motion comes from the edit instead: `index.html` pans over the 2× captures, which stays sharp at 1080p.

## Rebuild

Same tools as the [promo](../promo/README.md). The promo's `render.mjs` and `narrate.py` render this folder with `VIDEO=chapter-6-1`:

```bash
export VIDEO=chapter-6-1
node video/chapter-6-1/capture.mjs                              # shots/ from the live site (SITE=... to use another)
node video/promo/render.mjs --meta                              # out/events.json + out/timeline.json
~/.cache/kokoro-tts/venv/bin/python video/promo/narrate.py      # out/narration/*.wav
python3 video/chapter-6-1/soundtrack.py                         # out/soundtrack.wav
node video/promo/render.mjs video/chapter-6-1/out/silent.mp4
ffmpeg -i video/chapter-6-1/out/silent.mp4 -i video/chapter-6-1/out/soundtrack.wav \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart video/chapter-6-1/out/chapter-6-1.mp4
ffmpeg -i video/chapter-6-1/out/chapter-6-1.mp4 -c:v libx264 -preset slow -crf 28 -c:a aac -b:a 128k \
  -movflags +faststart public/video/chapter-6-1.mp4               # the copy on the home page (~5 MB)
```

If the chapter's layout changes, re-run `capture.mjs`: highlights and clicks follow the new positions from `shots.json`.
