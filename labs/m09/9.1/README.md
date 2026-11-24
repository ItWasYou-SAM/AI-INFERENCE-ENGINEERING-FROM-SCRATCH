# Lab 9.1 — How big is the KV cache? 🟢 CPU

```bash
pip install numpy
python labs/m09/9.1/kv_cache.py
```

**Exercises**
1. Add Llama 3.1 8B (32 layers, 8 KV heads, head dim 128, 8.03 B params). How many users fit on an L4 (24 GB) at 4K?
2. At what context length does a single Mistral 7B user's KV cache exceed the model's own FP16 weights?
3. Change `measure_with_numpy` to store K and V in `float8`-sized `uint8` arrays and confirm that the bytes halve.
