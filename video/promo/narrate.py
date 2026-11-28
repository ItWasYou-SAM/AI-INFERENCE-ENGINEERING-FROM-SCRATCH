"""Speak the promo's narration with Kokoro, a small open TTS model that runs offline on the CPU.

    python video/promo/narrate.py        out/timeline.json -> out/narration/NN.wav + out/narration/lines.json

Run it with the Python that has kokoro-onnx installed (see README). The lines and their start times
come from NARRATION in index.html, via `render.mjs --meta`. A line that would run into the next one
is spoken a little faster (up to MAX_SPEED); if it still does not fit, the script says so and exits 1.

Env: VIDEO (default promo: which folder under video/ to narrate, e.g. matmul),
     KOKORO_DIR (default ~/.cache/kokoro-tts, holds kokoro-v1.0.onnx and voices-v1.0.bin),
     VOICE (default af_heart), SPEED (default 1.0).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent.parent / os.environ.get('VIDEO', 'promo')
OUT = HERE / 'out' / 'narration'
KDIR = Path(os.environ.get('KOKORO_DIR', Path.home() / '.cache' / 'kokoro-tts'))
VOICE = os.environ.get('VOICE', 'af_heart')
SPEED = float(os.environ.get('SPEED', '1.0'))
MAX_SPEED = 1.12
GAP = 0.15  # seconds of air before the next line


def trim(x, sr, thresh=0.01):
    """Cut the model's leading and trailing silence, keeping a few ms so consonants are not clipped."""
    idx = np.flatnonzero(np.abs(x) > thresh)
    if len(idx) == 0:
        return x
    pad = int(0.03 * sr)
    return x[max(0, idx[0] - pad): idx[-1] + pad]


def main():
    tl = json.loads((HERE / 'out' / 'timeline.json').read_text())
    lines, dur = tl['narration'], tl['duration']
    kokoro = Kokoro(str(KDIR / 'kokoro-v1.0.onnx'), str(KDIR / 'voices-v1.0.bin'))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, ok = [], True
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]['t'] if i + 1 < len(lines) else dur - 0.6
        slot = nxt - ln['t'] - GAP
        speed = SPEED
        while True:
            audio, sr = kokoro.create(ln['text'], voice=VOICE, speed=speed, lang='en-us')
            audio = trim(np.asarray(audio, dtype=np.float32), sr)
            length = len(audio) / sr
            if length <= slot or speed >= MAX_SPEED:
                break
            speed = min(MAX_SPEED, speed * min(1.1, length / slot + 0.01))
        f = OUT / f'{i:02d}.wav'
        sf.write(f, audio, sr)
        fits = length <= slot
        ok &= fits
        print(f'{ln["t"]:5.2f}s  {length:4.2f}s of {slot:4.2f}s  x{speed:.2f}  {"ok " if fits else "TOO LONG"}  {ln["text"]}')
        manifest.append({'t': ln['t'], 'file': f.name, 'sr': sr, 'length': round(length, 3), 'text': ln['text']})
    (OUT / 'lines.json').write_text(json.dumps(manifest, indent=2))
    if not ok:
        sys.exit('some lines do not fit: shorten them or move the scene times (TL) in index.html')


if __name__ == '__main__':
    main()
