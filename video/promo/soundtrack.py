"""Synthesize the promo soundtrack and mix in the narrator.

    python3 video/promo/soundtrack.py      -> video/promo/out/soundtrack.wav

Reads out/events.json and out/timeline.json (written by render.mjs) and, if present, the spoken
lines from out/narration/ (written by narrate.py). The music ducks under the voice.

Reuses the trailer's synthesizer (video/trailer/soundtrack.py): same pad, plucks,
hits and whooshes, all generated with NumPy. Only the length, chords and mix differ.
"""
import importlib.util
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('trailer_sound', HERE.parent / 'trailer' / 'soundtrack.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

TL = json.loads((HERE / 'out' / 'timeline.json').read_text())
SC = TL['scenes']
DUR = float(TL['duration'])
SR = S.SR
N = int(SR * DUR)
# the trailer's pad() reads these module globals
S.DUR, S.N, S.t_all = DUR, N, np.arange(N) / SR
S.SECTIONS = [
    (SC['hook'], [45, 52], 0.5),                    # A, sparse: the question
    (SC['course'], [41, 48, 57, 64], 1.0),          # Fmaj7: the reveal
    (SC['roadmap'], [45, 52, 60, 67], 0.85),        # Am7: roadmap
    (SC['req'], [38, 45, 53, 60, 64], 0.85),        # Dm9: figures
    (SC['kv'], [41, 48, 57, 64], 0.85),             # Fmaj7
    (SC['labs'], [48, 55, 59, 64], 0.85),           # Cmaj7: labs
    (SC['cta'], [48, 55, 64, 67, 72], 1.1),         # C: resolve
]


def narration():
    """The spoken lines placed on the timeline (mono, peak 1), or None if narrate.py has not run."""
    f = HERE / 'out' / 'narration' / 'lines.json'
    if not f.exists():
        return None
    voice = np.zeros(N)
    for ln in json.loads(f.read_text()):
        with wave.open(str(f.parent / ln['file'])) as w:
            sr, raw = w.getframerate(), w.readframes(w.getnframes())
            x = np.frombuffer(raw, '<i2').astype(np.float64) / 32768 if w.getsampwidth() == 2 else None
        if x is None:  # soundfile writes 16-bit PCM by default; anything else is unexpected
            raise SystemExit(f'{ln["file"]}: expected 16-bit PCM')
        x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x)
        i0 = int(ln['t'] * SR)
        n = min(len(x), N - i0)
        voice[i0:i0 + n] += x[:n]
    return voice / max(1e-9, np.max(np.abs(voice)))


def main():
    events = json.loads((HERE / 'out' / 'events.json').read_text())
    fx = np.zeros((2, N))
    for e in events:
        s = S.SHOTS[e['type']](e['p'])
        i0 = int(e['t'] * SR)
        if i0 >= N:
            continue
        s = s[: min(N, i0 + len(s)) - i0]
        spread = S.PAN.get(e['type'], 0.0)
        pan = spread * np.sin(e['p'] * 1.7 + e['t'])
        fx[0, i0:i0 + len(s)] += s * np.sqrt((1 - pan) / 2) * 1.414
        fx[1, i0:i0 + len(s)] += s * np.sqrt((1 + pan) / 2) * 1.414

    # a steady pulse under the middle of the video
    beat = np.zeros(N)
    k = S.boom(0)[: int(0.5 * SR)] * S.env_exp(int(0.5 * SR), 0.12, 0.002) * 0.35
    for bt in np.arange(SC['req'], SC['cta'] - 0.4, 60 / 84):
        i0 = int(bt * SR)
        beat[i0:i0 + len(k)] += k[: N - i0]
    fx += beat

    ir_len = int(2.6 * SR)
    tt = np.arange(ir_len) / SR
    wet = np.zeros_like(fx)
    L = 1 << int(np.ceil(np.log2(N + ir_len)))
    for ch in range(2):
        ir = S.rng.standard_normal(ir_len) * np.exp(-tt / 0.55)
        ir = S.lowpass(ir, 5000)
        ir /= np.sqrt(np.sum(ir ** 2))
        wet[ch] = np.fft.irfft(np.fft.rfft(fx[ch], L) * np.fft.rfft(ir, L), L)[:N]

    mix = S.pad() + fx * 0.8 + wet * 0.35
    t_all = np.arange(N) / SR
    mix *= np.clip(t_all / 0.5, 0, 1) * np.clip((DUR - 0.1 - t_all) / 1.4, 0, 1)
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix *= 0.89 / np.max(np.abs(mix))

    voice = narration()
    if voice is not None:
        # duck the music while the narrator speaks: a smoothed envelope of the voice, ~80 ms attack, ~400 ms release
        env = np.abs(voice) > 0.02
        hop = SR // 100
        frames = env[: len(env) // hop * hop].reshape(-1, hop).any(1).astype(float)
        sm = np.zeros_like(frames)
        for i in range(len(frames)):
            prev = sm[i - 1] if i else 0.0
            k = 0.12 if frames[i] > prev else 0.025
            sm[i] = prev + (frames[i] - prev) * k
        duck = 1 - 0.62 * np.interp(np.arange(N), np.arange(len(sm)) * hop, sm)
        mix = mix * 0.85 * duck + voice * 0.92
        mix = np.tanh(mix * 1.1) / np.tanh(1.1)
        mix *= 0.95 / np.max(np.abs(mix))

    out = HERE / 'out' / 'soundtrack.wav'
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype('<i2').tobytes())
    print('wrote', out)


if __name__ == '__main__':
    main()
