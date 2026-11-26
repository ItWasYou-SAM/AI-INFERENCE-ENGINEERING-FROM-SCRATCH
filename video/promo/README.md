# Promo

A 28-second, 1080p narrated promo for students, built from **real screenshots of the running site**, plus six promo images.
Same look and synthesized sound as the [trailer](../trailer/README.md), which explains the ideas; this one shows the course working.
A narrator (Kokoro TTS, generated offline) talks over it and the music ducks under the voice.

| Time | Scene | Narrator |
|:--|:--|:--|
| 0:00 | Hook: "What happens when you press Enter?" | "You talk to an AI every day. What happens when you press Enter?" |
| 0:04 | The course: home page, light then dark | "Inference Engineering, from Scratch: a free course on how LLMs are served." |
| 0:08 | Roadmap: 4 levels, 18 modules, 109 chapters | "Four levels take you from your very first token to serving at scale." |
| 0:12 | Figure 0.2: one request through a server | "Every idea is interactive. Follow one request through a server," |
| 0:16 | Figure 9.1: KV calculator, Fits → OOM → FP8 → Fits | "then add users until the GPU runs out of memory." |
| 0:19.6 | Labs: the 5.7 softmax lab, real RTX 3050 output | "Write real GPU kernels, and measure them yourself." |
| 0:23.6 | Call to action: GitHub link, Chapter 0.1 | "Free and open source. Start with chapter zero point one." |

Scene times (`TL`) and the narration lines (`NARRATION`) are both at the top of the script in `index.html`.

## Images (`out/images/`)

| File | For |
|:--|:--|
| `og-card-1200x630.png` | GitHub social preview, link previews |
| `thumbnail-1280x720.png` | YouTube thumbnail |
| `post-square-1080x1080.png` | Instagram / LinkedIn post |
| `story-1080x1920.png` | Instagram / WhatsApp story, printable poster |
| `use-cases-1080x1350.png` | Instagram portrait / carousel slide |
| `banner-1500x500.png` | X / LinkedIn banner |

## Rebuild

Needs Chromium, ffmpeg, Python 3 with NumPy, and `puppeteer-core` installed somewhere (`PPTR_HOME`).
The narrator needs [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) in its own venv and its model files
(about 340 MB, `KOKORO_DIR`, default `~/.cache/kokoro-tts`):

```bash
python3 -m venv ~/.cache/kokoro-tts/venv && ~/.cache/kokoro-tts/venv/bin/pip install kokoro-onnx soundfile
(cd ~/.cache/kokoro-tts && curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
                         -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)
```

```bash
npm run build && npx astro preview --port 4399 &            # the site to capture
node video/promo/capture.mjs                                  # → shots/*.jpg (real site, 1440x900 @2x)
node video/promo/render.mjs video/promo/out/silent.mp4        # → video frames, out/events.json, out/timeline.json
~/.cache/kokoro-tts/venv/bin/python video/promo/narrate.py    # → out/narration/*.wav (VOICE=af_heart by default)
python3 video/promo/soundtrack.py                             # → out/soundtrack.wav, music + narrator
ffmpeg -i video/promo/out/silent.mp4 -i video/promo/out/soundtrack.wav \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart video/promo/out/promo.mp4
node video/promo/render.mjs --meta                            # only events.json + timeline.json (to redo the audio)
node video/promo/render.mjs --images                          # → out/images/*.png
node video/promo/render.mjs --stills 2,10,20                  # quick PNG stills
```

`narrate.py` speeds a line up slightly if it would run into the next one, and fails if it still does not fit.

Preview in a browser by serving `video/` (for example `npx serve video`) and opening `/promo/index.html`
(`?t=20` jumps to a second) or `/promo/images.html`.
