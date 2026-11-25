# AI Inference Engineering — From Scratch: Roadmap

> Generated from `src/data/roadmap.ts` by `npm run roadmap`. Do not edit by hand.

4 levels · 18 modules · 109 chapters.

Each chapter follows the same arc: **Concept → Animation / interactive → Code lab → Sources → Quiz**.

Lab tags: 🟢 CPU or browser · 🟡 GPU optional · 🔴 GPU required (Colab or a rented GPU) · ✨ interactive visual

## Sources

| ID | Source | Kind |
|---|---|---|
| `BGE` | [BEFORE-GPU-ENGINEERING](https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING) — BEFORE-GPU-ENGINEERING | repo |
| `TTFT` | [time-to-first-token](https://github.com/patchy631/time-to-first-token) — patchy631 | course |
| `CUDA100` | [100 Days of CUDA](https://pavelsimo.github.io/100-days-of-cuda/) — Pavel Simo | course |
| `ROUTER` | [vLLM Router release](https://vllm.ai/blog/2025-12-13-vllm-router-release) — vLLM Team | blog |
| `MOHITH` | [LLM Inference at Scale](https://www.mohith.xyz/blog/llm-inference-at-scale) — Mohith | blog |
| `XPOST` | [Akshay Pachaar on X](https://x.com/akshay_pachaar/status/2020840784782913605) — Akshay Pachaar *(placeholder)* | post |
| `ICP` | [From Inference Engine to Inference Control Plane](https://arxiv.org/abs/2609.23130v1) — Twinkll Sisodia | paper |
| `FREGLY` | *AI Systems Performance Engineering* — Chris Fregly | book |

## Level 1 — Beginner: "What is inference, and why is it hard?"

### M0 · Orientation

_A map of the territory: what inference engineers do, and how a request flows._

**Sources:** `MOHITH`, `TTFT` (Overview)

- **0.1** What inference engineering is ✨
- **0.2** The life of a request: tokens in, tokens out 🟡 ✨
- **0.3** How to use this site and the labs 🟢

### M1 · The LLM forward pass, for inference engineers

_Only the transformer you need in order to reason about cost._

**Sources:** `MOHITH` (Pipeline, layer, attention), `CUDA100` (Days 14, 28, 31)

- **1.1** Tokens, embeddings and logits
- **1.2** Inside a transformer layer ✨
- **1.3** Attention, step by step ✨
- **1.4** Autoregressive generation and sampling basics
- **1.5** The KV cache: why it exists ✨
- **1.6** Lab: a tiny GPT inference loop in NumPy 🟢

### M2 · Why GPUs: hardware first principles

_Why the hardware looks the way it does, before writing any CUDA._

**Sources:** `BGE` (Phase 1, Ch 1–15), `FREGLY` (GPU architecture chapters)

- **2.1** Why CPUs can't have 10,000 cores
- **2.2** Parallelism and what "parallelizable" means 🟢
- **2.3** Memory bandwidth and the memory hierarchy 🟢 ✨
- **2.4** Warps, SIMT and branching ✨
- **2.5** Occupancy and latency hiding 🟢 ✨
- **2.6** The memory wall and HBM
- **2.7** Thought experiment: designing an LLM processor

### M3 · The performance mental model

_Roofline, arithmetic intensity, and why decode is memory-bound._

**Sources:** `BGE` (Ch 33–35), `TTFT` (Week 1), `MOHITH` (Two phases of inference), `FREGLY` (Arithmetic intensity chapter)

- **3.1** Compute-bound vs memory-bound
- **3.2** Arithmetic intensity
- **3.3** The roofline model 🟢 ✨
- **3.4** Prefill vs decode: TTFT and TPOT ✨
- **3.5** Lab: derive the arithmetic intensity of a 7B model 🟢

## Level 2 — Intermediate: "How do I write and optimize GPU code?"

### M4 · CUDA programming fundamentals

_Kernels, threads, blocks and memory: the CUDA programming model._

**Sources:** `BGE` (Phase 2, Ch 16–32), `CUDA100` (Days 1–7)

- **4.1** A kernel is not a function 🔴
- **4.2** Thread, block and grid indexing 🔴 ✨
- **4.3** Host/device memory and transfers 🔴
- **4.4** Shared memory and __syncthreads 🔴
- **4.5** Race conditions and correctness 🔴
- **4.6** Kernel launch anatomy, async execution and streams 🔴 ✨
- **4.7** Coalesced vs strided access 🔴 ✨
- **4.8** When not to use a GPU

### M5 · Parallel patterns

_Reduction, scan, histogram and convolution: the building blocks of every kernel._

**Sources:** `CUDA100` (Days 8–25), `BGE` (Ch 32, 47)

- **5.1** Reduction trees 🔴 ✨
- **5.2** Warp shuffles 🔴
- **5.3** Atomics and compare-and-swap 🔴
- **5.4** Prefix sum (scan) 🔴 ✨
- **5.5** Histograms and privatization 🔴
- **5.6** Convolution and stencils 🔴
- **5.7** Softmax: three kernels, then one fused kernel 🔴

### M6 · GEMM and memory optimization

_From a naive matmul to a tiled, tensor-core-aware kernel._

**Sources:** `BGE` (Phase 3, Ch 36–51), `CUDA100` (Days 15, 33), `FREGLY` (Memory access, occupancy and ILP chapters)

- **6.1** Naive → coalesced → shared-memory tiled matmul 🔴 ✨
- **6.2** Register tiling: the tile inside the tile 🔴
- **6.3** Register pressure, spilling and the occupancy trade-off
- **6.4** Shared-memory bank conflicts ✨
- **6.5** Wave quantization ✨
- **6.6** Latency hiding and instruction-level parallelism
- **6.7** Tensor cores and mixed precision 🔴
- **6.8** Reading PTX and SASS 🔴

### M7 · Profiling: measure, don't guess

_Nsight, the PyTorch profiler, and a top-down profiling discipline._

**Sources:** `BGE` (Phase 4, Ch 52–56), `FREGLY` (PyTorch profiling chapter)

- **7.1** The top-down profiling discipline
- **7.2** Nsight Systems vs Nsight Compute 🔴
- **7.3** What kernel time actually measures
- **7.4** The PyTorch profiler and traces 🟡
- **7.5** Case study: the 91 missing TFLOPS 🔴

### M8 · Transformer kernels and compilers

_The kernels an LLM actually runs, plus Triton and torch.compile._

**Sources:** `CUDA100` (Days 14, 20, 26–28, 31, 34, 38–41, 77), `FREGLY` (torch.compile / Triton chapter)

- **8.1** RMSNorm, LayerNorm and fused residual add 🔴
- **8.2** RoPE and ALiBi 🔴 ✨
- **8.3** The SwiGLU MLP block 🔴
- **8.4** A multi-head attention kernel 🔴
- **8.5** FlashAttention: tiling and online softmax 🟡 ✨
- **8.6** Triton for kernel authors 🔴
- **8.7** torch.compile and CUDA Graphs 🔴
- **8.8** Fused QKV + RoPE + KV-cache update 🔴

## Level 3 — Advanced: "How do I build and tune a serving engine?"

### M9 · KV cache engineering

_The memory that decides how many users you can serve._

**Sources:** `MOHITH` (KV size, memory budget, capacity equation), `CUDA100` (Days 35, 47, 74, 93, 100)

- **9.1** How big is the KV cache? The capacity equation 🟢 ✨
- **9.2** MQA, GQA and MLA ✨
- **9.3** PagedAttention and block tables 🟢 ✨
- **9.4** Variable-length batches (cu_seqlens) 🟡
- **9.5** Split-KV and FlashDecoding ✨
- **9.6** KV-cache quantization 🟡
- **9.7** Sliding windows, attention sinks and eviction ✨

### M10 · Serving engine internals

_Schedulers, batching and prefix caching inside vLLM and SGLang._

**Sources:** `TTFT` (Weeks 2 and 4), `MOHITH` (Serving-level optimizations), `FREGLY` (Dynamic scheduling chapter)

- **10.1** Static vs continuous batching ✨
- **10.2** The scheduler: admission, preemption and swapping ✨
- **10.3** Chunked prefill (Sarathi-Serve) ✨
- **10.4** Prefix caching and RadixAttention ✨
- **10.5** A walkthrough of the vLLM V1 architecture
- **10.6** SGLang vs vLLM
- **10.7** Lab: deploy a 7–8B model behind an OpenAI-compatible API 🔴
- **10.8** Lab: build a mini engine (paged KV + continuous batching) 🟢

### M11 · Decoding and sampling

_How the next token is chosen, and how to choose several at once._

**Sources:** `CUDA100` (Days 68, 75, 79, 81), `TTFT` (Week 7)

- **11.1** Greedy, top-k, top-p and min-p 🟢 ✨
- **11.2** Beam search
- **11.3** Speculative decoding: draft, verify, accept ✨
- **11.4** EAGLE- and Medusa-style variants
- **11.5** When speculation stops paying off: the QPS crossover 🔴

### M12 · Quantization for inference

_Fewer bits, fewer bytes moved, more tokens per second._

**Sources:** `TTFT` (Week 6), `CUDA100` (Days 29, 51, 53), `MOHITH` (Weight quantization), `FREGLY`

- **12.1** Number formats: FP32 → BF16 → FP8 → INT4 ✨
- **12.2** Weight-only vs weight + activation quantization
- **12.3** GPTQ and AWQ
- **12.4** A W4A16 matmul kernel 🔴
- **12.5** Serving in FP8 🔴
- **12.6** Lab: FP16 vs FP8 vs INT4 throughput and quality 🔴

### M13 · Measurement and benchmarking

_TTFT, ITL, goodput and tail latency, measured honestly._

**Sources:** `TTFT` (Weeks 3 and 5)

- **13.1** Metrics: TTFT, ITL, TPOT, throughput and goodput ✨
- **13.2** Percentiles and tail latency ✨
- **13.3** Prometheus and Grafana for vLLM 🔴
- **13.4** Load testing with GuideLLM and vllm bench serve 🔴
- **13.5** Benchmarking pitfalls and reproducibility

## Level 4 — Expert: "How do I run inference at scale?"

### M14 · Multi-GPU inference and parallelism

_Splitting a model across GPUs, and what that costs in communication._

**Sources:** `FREGLY` (Distributed networking and multinode chapters), `CUDA100` (Days 67, 92)

- **14.1** Collectives and NCCL ✨
- **14.2** Tensor parallelism ✨
- **14.3** Pipeline parallelism ✨
- **14.4** Data parallelism and replicas
- **14.5** MoE: gating, grouped GEMM and expert parallelism ✨
- **14.6** Interconnects: NVLink, InfiniBand and RDMA

### M15 · Distributed serving and routing

_Prefill/decode disaggregation, KV transfer, and smart routers._

**Sources:** `ROUTER`, `TTFT` (Week 8), `ICP`, `FREGLY` (Disaggregated prefill/decode chapters)

- **15.1** Why disaggregate prefill and decode ✨
- **15.2** KV transfer: NIXL and NCCL
- **15.3** Load balancing: round robin, power of two, KV affinity ✨
- **15.4** Inside the vLLM Router
- **15.5** Resiliency: retries and circuit breakers
- **15.6** Kubernetes, the production stack and queue-depth autoscaling 🔴

### M16 · Economics and the inference control plane

_Cost per token, SLO goodput, and the move from engine to control plane._

**Sources:** `TTFT` (Weeks 9–10), `ICP`, `XPOST`

- **16.1** Cost per million tokens 🟢 ✨
- **16.2** SLOs and goodput
- **16.3** Cost-aware routing
- **16.4** From engine to control plane: llm-d and the execution planner
- **16.5** Bottleneck migration
- **16.6** Reading the frontier: arXiv cs.DC, MLSys and OSDI

### Capstone · Capstone: ship one service

_Deploy, instrument, load-test, tune, route, and publish a reproducible benchmark._

**Sources:** `TTFT` (The whole 10-week path)

- **C.1** Deploy and instrument 🔴
- **C.2** Load-test, quantize and speculate 🔴
- **C.3** Route and publish the benchmark 🔴
