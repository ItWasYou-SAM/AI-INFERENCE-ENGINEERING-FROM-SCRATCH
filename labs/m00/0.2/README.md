# Lab 0.2 — Measure TTFT and TPOT 🟡 GPU optional

Start any OpenAI-compatible server. Ollama or llama.cpp run on a CPU with a 0.5B model. Then:

```bash
python labs/m00/0.2/measure_ttft.py --base http://localhost:11434 --model qwen2.5:0.5b
```

**Exercises**
1. Make the prompt 10× longer (paste a paragraph). Which number moves, TTFT or TPOT? Why?
2. Raise `--max-tokens` to 512. Does TPOT change as the output grows?
3. Run two copies at once. What happens to each one's TTFT and TPOT?
