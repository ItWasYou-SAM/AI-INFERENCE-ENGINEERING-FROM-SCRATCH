"""
Lab 0.2 — Measure TTFT and TPOT of a real request

Streams one chat completion from any OpenAI-compatible server and times
  * TTFT: request sent -> first content token received
  * TPOT: average gap between the following tokens
Standard library only.

Servers you can point it at:
  vLLM (GPU):   vllm serve Qwen/Qwen2.5-0.5B-Instruct          -> http://localhost:8000
  Ollama (CPU): ollama run qwen2.5:0.5b                        -> http://localhost:11434
  llama.cpp:    llama-server -m model.gguf                     -> http://localhost:8080

Run:
  python labs/m00/0.2/measure_ttft.py --base http://localhost:8000 --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import time
import urllib.request


def stream_chat(base: str, model: str, prompt: str, max_tokens: int):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        for raw in resp:  # server-sent events: lines of "data: {...}"
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}).get("content") if choices else None
            if delta:
                yield delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Explain the KV cache in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()

    t0 = time.perf_counter()
    stamps, text = [], []
    for piece in stream_chat(a.base, a.model, a.prompt, a.max_tokens):
        stamps.append(time.perf_counter())
        text.append(piece)
        print(piece, end="", flush=True)
    print("\n")

    if not stamps:
        print("No tokens received.")
        return
    ttft = (stamps[0] - t0) * 1000
    gaps = [(b - a_) * 1000 for a_, b in zip(stamps, stamps[1:])]
    tpot = sum(gaps) / len(gaps) if gaps else float("nan")
    total = (stamps[-1] - t0) * 1000
    print(f"chunks received : {len(stamps)}")
    print(f"TTFT            : {ttft:8.1f} ms")
    print(f"TPOT (mean gap) : {tpot:8.1f} ms  (~{1000 / tpot:.0f} tokens/s)" if gaps else "TPOT: n/a")
    print(f"end-to-end      : {total:8.1f} ms  = TTFT + (n-1) x TPOT")
    print("\nNote: servers may batch several tokens into one chunk, so 'chunks' can be fewer than tokens.")


if __name__ == "__main__":
    main()
