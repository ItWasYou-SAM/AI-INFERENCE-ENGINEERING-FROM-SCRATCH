"""Synthesize the trailer soundtrack from out/events.json (written by render.mjs).

    python3 video/trailer/soundtrack.py      -> out/soundtrack.wav

Everything is generated with NumPy: a slow pad that follows the scenes, plucks on
every generated token, and a few cinematic hits and whooshes. No samples, no licences.
"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SR = 48000
DUR = 80.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
t_all = np.arange(N) / SR


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lowpass(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def bandpass(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= (1 / np.sqrt(1 + (f / hi) ** 4)) * (1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4))
    return np.fft.irfft(X, len(x))


# ── pad ──────────────────────────────────────────────────────────────
SECTIONS = [  # (start, chord as MIDI notes, level)
    (0, [45, 52], 0.5),                 # A  (cold open, sparse)
    (8, [41, 48, 57, 64], 1.0),         # Fmaj7
    (14, [45, 52, 60, 67], 0.8),        # Am7
    (22, [38, 45, 53, 60, 64], 0.8),    # Dm9
    (32, [41, 48, 57, 64], 0.8),        # Fmaj7
    (40, [48, 55, 59, 64], 0.8),        # Cmaj7
    (50, [43, 50, 57, 60], 0.8),        # Gsus
    (59, [45, 52, 60, 67], 0.8),        # Am7
    (66, [41, 48, 57, 64, 67], 0.9),    # Fmaj9
    (73, [48, 55, 64, 67, 72], 1.1),    # C  (resolve)
]


def pad():
    out = np.zeros((2, N))
    xf = 1.6  # crossfade seconds
    for i, (s, notes, lvl) in enumerate(SECTIONS):
        e = SECTIONS[i + 1][0] if i + 1 < len(SECTIONS) else DUR
        a, b = max(0, s - xf / 2), min(DUR, e + xf / 2)
        i0, i1 = int(a * SR), int(b * SR)
        tt = t_all[i0:i1]
        env = np.ones_like(tt)
        ramp = int(xf * SR)
        if i > 0:
            env[:ramp] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
        if i + 1 < len(SECTIONS):
            env[-ramp:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, ramp))
        for ch, det in ((0, -0.08), (1, 0.08)):
            sig = np.zeros_like(tt)
            for k, m in enumerate(notes):
                f0 = midi(m) * 2 ** (det / 12)
                ph = rng.uniform(0, 2 * np.pi)
                for h in range(1, 7):
                    sig += np.sin(2 * np.pi * f0 * h * tt + ph * h) / h ** 1.6
                sig *= 1.0  # keep
            lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.11 * tt + ch)
            out[ch, i0:i1] += sig * env * lvl * lfo / len(notes)
    for ch in range(2):
        out[ch] = lowpass(out[ch], 900)
    return out * 0.16


# ── one-shots ────────────────────────────────────────────────────────
PENTA = [81, 84, 86, 88, 91, 93]


def env_exp(n, decay, attack=0.004):
    tt = np.arange(n) / SR
    a = np.clip(tt / attack, 0, 1)
    return a * np.exp(-tt / decay)


def tick(p):
    n = int(0.6 * SR); tt = np.arange(n) / SR
    f = midi(PENTA[p % len(PENTA)])
    s = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt) + 0.1 * np.sin(2 * np.pi * 3.01 * f * tt)
    return s * env_exp(n, 0.16) * 0.22


def blip(p):
    n = int(0.4 * SR); tt = np.arange(n) / SR
    f = midi(PENTA[p % len(PENTA)] - 12)
    s = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 3 * f * tt)
    return s * env_exp(n, 0.09) * 0.16


def key(_):
    n = int(0.03 * SR)
    s = bandpass(rng.standard_normal(n), 1500, 6000)
    return s * env_exp(n, 0.006, 0.0005) * 0.12


def boom(_):
    n = int(3.5 * SR); tt = np.arange(n) / SR
    f = 32 + 30 * np.exp(-tt / 0.25)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 1.1, 0.01)
    hit = lowpass(rng.standard_normal(n), 400) * env_exp(n, 0.25, 0.002) * 0.6
    shimmer = sum(np.sin(2 * np.pi * midi(m) * tt) for m in (76, 81, 88)) * env_exp(n, 1.4, 0.4) * 0.05
    return (sub * 0.9 + hit + shimmer) * 0.9


def whoosh(_, length=1.1, lo=300, hi=3000, gain=0.22):
    n = int(length * SR); tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    a = bandpass(noise, lo, lo * 3)
    b = bandpass(noise, hi / 3, hi)
    x = tt / length
    mix = a * (1 - x) + b * x
    shape = np.sin(np.pi * np.clip(x, 0, 1)) ** 2
    return mix * shape * gain


def swish(p):
    return whoosh(p, 0.5, 800, 6000, 0.14)


def alarm(_):
    n = int(0.9 * SR); tt = np.arange(n) / SR
    s = np.sign(np.sin(2 * np.pi * midi(57) * tt)) * 0.5 + np.sign(np.sin(2 * np.pi * midi(58) * tt)) * 0.5
    s = lowpass(s, 1800)
    gate = (np.sin(2 * np.pi * 7 * tt) > 0).astype(float)
    return s * gate * env_exp(n, 0.5) * 0.1


def chime(_):
    n = int(3.0 * SR); tt = np.arange(n) / SR
    s = np.zeros(n)
    for root in (76, 83):
        f = midi(root)
        for ratio, amp, dec in ((1, 1, 1.4), (2.76, 0.4, 0.6), (5.4, 0.2, 0.3)):
            s += amp * np.sin(2 * np.pi * f * ratio * tt) * env_exp(n, dec, 0.002)
    return s * 0.09


SHOTS = dict(tick=tick, blip=blip, key=key, boom=boom, whoosh=whoosh, swish=swish, alarm=alarm, chime=chime)
PAN = dict(tick=0.35, blip=0.5, key=0.2)


def main():
    events = json.loads((HERE / 'out' / 'events.json').read_text())
    fx = np.zeros((2, N))
    for e in events:
        s = SHOTS[e['type']](e['p'])
        i0 = int(e['t'] * SR)
        i1 = min(N, i0 + len(s))
        s = s[: i1 - i0]
        spread = PAN.get(e['type'], 0.0)
        pan = spread * np.sin(e['p'] * 1.7 + e['t'])  # -spread..spread, deterministic
        fx[0, i0:i1] += s * np.sqrt((1 - pan) / 2) * 1.414
        fx[1, i0:i1] += s * np.sqrt((1 + pan) / 2) * 1.414

    # gentle heartbeat under the explainer scenes
    beat = np.zeros(N)
    k = boom(0)[: int(0.5 * SR)] * env_exp(int(0.5 * SR), 0.12, 0.002) * 0.35
    for bt in np.arange(14.0, 66.0, 60 / 72):
        i0 = int(bt * SR); beat[i0:i0 + len(k)] += k[: N - i0]
    fx += beat

    # reverb: decorrelated exponential noise tails
    ir_len = int(2.6 * SR)
    tt = np.arange(ir_len) / SR
    wet = np.zeros_like(fx)
    L = 1 << int(np.ceil(np.log2(N + ir_len)))
    for ch in range(2):
        ir = rng.standard_normal(ir_len) * np.exp(-tt / 0.55)
        ir = lowpass(ir, 5000); ir /= np.sqrt(np.sum(ir ** 2))
        wet[ch] = np.fft.irfft(np.fft.rfft(fx[ch], L) * np.fft.rfft(ir, L), L)[:N]

    mix = pad() + fx * 0.8 + wet * 0.35
    fade = np.clip(t_all / 0.6, 0, 1) * np.clip((DUR - 0.2 - t_all) / 1.6, 0, 1)
    mix *= fade
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    mix *= 0.89 / np.max(np.abs(mix))

    pcm = (mix.T * 32767).astype('<i2')
    out = HERE / 'out' / 'soundtrack.wav'
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('wrote', out)


if __name__ == '__main__':
    main()
