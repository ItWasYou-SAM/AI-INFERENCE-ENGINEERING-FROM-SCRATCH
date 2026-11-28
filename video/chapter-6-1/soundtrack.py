"""Synthesize the chapter 6.1 walk-through's soundtrack and mix in the narrator.

    python3 video/chapter-6-1/soundtrack.py      -> video/chapter-6-1/out/soundtrack.wav

Reads out/events.json and out/timeline.json (from `VIDEO=chapter-6-1 render.mjs`) and, if present, the
spoken lines in out/narration/ (from `VIDEO=chapter-6-1 narrate.py`). Same NumPy synthesizer as the
trailer; the pulse speeds up through the chapter and the music ducks under the voice.
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
S.DUR, S.N, S.t_all = DUR, N, np.arange(N) / SR
S.SECTIONS = [
    (SC['hook'], [45, 52, 60], 0.6),               # Am, sparse: the 60× gap
    (SC['code'], [38, 45, 53, 60, 64], 0.85),  # Dm9
    (SC['figure'], [41, 48, 57, 64], 0.9),           # Fmaj7
    (SC['measured'], [43, 50, 59, 62, 67], 1.05),  # G: lift
    (SC['cta'], [48, 55, 64, 67, 72], 1.1),        # C: resolve
]


def narration():
    f = HERE / 'out' / 'narration' / 'lines.json'
    if not f.exists():
        return None
    voice = np.zeros(N)
    for ln in json.loads(f.read_text()):
        with wave.open(str(f.parent / ln['file'])) as w:
            if w.getsampwidth() != 2:
                raise SystemExit(f'{ln["file"]}: expected 16-bit PCM')
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
        x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x)
        i0 = int(ln['t'] * SR)
        n = min(len(x), N - i0)
        voice[i0:i0 + n] += x[:n]
    return voice / max(1e-9, np.max(np.abs(voice)))


def main():
    fx = np.zeros((2, N))
    for e in json.loads((HERE / 'out' / 'events.json').read_text()):
        i0 = int(e['t'] * SR)
        if i0 >= N:
            continue
        s = S.SHOTS[e['type']](e['p'])
        s = s[: min(N, i0 + len(s)) - i0]
        if e['type'] == 'key':
            s = s * 0.6  # quiet
        pan = S.PAN.get(e['type'], 0.0) * np.sin(e['p'] * 1.7 + e['t'])
        fx[0, i0:i0 + len(s)] += s * np.sqrt((1 - pan) / 2) * 1.414
        fx[1, i0:i0 + len(s)] += s * np.sqrt((1 + pan) / 2) * 1.414

    # a pulse that quickens as the kernel gets faster: 70 -> 120 bpm, then stops for the reveal
    beat = np.zeros(N)
    k = S.boom(0)[: int(0.5 * SR)] * S.env_exp(int(0.5 * SR), 0.12, 0.002) * 0.38
    bt = SC['code']
    while bt < SC['measured'] - 0.3:
        i0 = int(bt * SR)
        beat[i0:i0 + len(k)] += k[: N - i0]
        frac = (bt - SC['code']) / (SC['measured'] - SC['code'])
        bt += 60 / (70 + 50 * frac)
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
        env = np.abs(voice) > 0.02
        hop = SR // 100
        frames = env[: len(env) // hop * hop].reshape(-1, hop).any(1).astype(float)
        sm = np.zeros_like(frames)
        for i in range(len(frames)):
            prev = sm[i - 1] if i else 0.0
            sm[i] = prev + (frames[i] - prev) * (0.12 if frames[i] > prev else 0.025)
        duck = 1 - 0.6 * np.interp(np.arange(N), np.arange(len(sm)) * hop, sm)
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
