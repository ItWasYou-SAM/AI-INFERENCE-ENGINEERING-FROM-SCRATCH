import type { SourceId } from './sources';

/** cpu = CPU or in-browser · gpu-optional · gpu-required (Colab / rented GPU) */
export type Lab = 'cpu' | 'gpu-optional' | 'gpu-required';

export interface SourceRef {
  id: SourceId;
  /** What part of the source this module draws on, e.g. "Ch 1–15" or "Days 8–13". */
  note?: string;
}

export interface Chapter {
  id: string; // "9.1"
  slug: string; // "kv-cache-size"
  title: string;
  lab?: Lab;
  interactive?: boolean;
}

export interface Module {
  id: string; // "m09"
  num: number;
  title: string;
  tagline: string;
  sources: SourceRef[];
  chapters: Chapter[];
}

export interface Level {
  num: number;
  name: string;
  question: string;
  modules: Module[];
}

const c = (id: string, slug: string, title: string, extra: Partial<Chapter> = {}): Chapter => ({
  id,
  slug,
  title,
  ...extra,
});

export const levels: Level[] = [
  {
    num: 1,
    name: 'Beginner',
    question: 'What is inference, and why is it hard?',
    modules: [
      {
        id: 'm00',
        num: 0,
        title: 'Orientation',
        tagline: 'A map of the territory: what inference engineers do, and how a request flows.',
        sources: [{ id: 'MOHITH' }, { id: 'TTFT', note: 'Overview' }],
        chapters: [
          c('0.1', 'what-is-inference-engineering', 'What inference engineering is', { interactive: true }),
          c('0.2', 'life-of-a-request', 'The life of a request: tokens in, tokens out', { interactive: true, lab: 'gpu-optional' }),
          c('0.3', 'how-to-use-this-site', 'How to use this site and the labs', { lab: 'cpu' }),
        ],
      },
      {
        id: 'm01',
        num: 1,
        title: 'The LLM forward pass, for inference engineers',
        tagline: 'Only the transformer you need in order to reason about cost.',
        sources: [{ id: 'MOHITH', note: 'Pipeline, layer, attention' }, { id: 'CUDA100', note: 'Days 14, 28, 31' }],
        chapters: [
          c('1.1', 'tokens-embeddings-logits', 'Tokens, embeddings and logits'),
          c('1.2', 'inside-a-transformer-layer', 'Inside a transformer layer', { interactive: true }),
          c('1.3', 'attention-step-by-step', 'Attention, step by step', { interactive: true }),
          c('1.4', 'autoregressive-generation', 'Autoregressive generation and sampling basics'),
          c('1.5', 'why-kv-cache', 'The KV cache: why it exists', { interactive: true }),
          c('1.6', 'tiny-gpt-numpy', 'Lab: a tiny GPT inference loop in NumPy', { lab: 'cpu' }),
        ],
      },
      {
        id: 'm02',
        num: 2,
        title: 'Why GPUs: hardware first principles',
        tagline: 'Why the hardware looks the way it does, before writing any CUDA.',
        sources: [{ id: 'BGE', note: 'Phase 1, Ch 1–15' }, { id: 'FREGLY', note: 'GPU architecture chapters' }],
        chapters: [
          c('2.1', 'why-not-10000-cpu-cores', "Why CPUs can't have 10,000 cores"),
          c('2.2', 'what-parallelizable-means', 'Parallelism and what "parallelizable" means', { lab: 'cpu' }),
          c('2.3', 'bandwidth-and-memory-hierarchy', 'Memory bandwidth and the memory hierarchy', { interactive: true, lab: 'cpu' }),
          c('2.4', 'warps-simt-branching', 'Warps, SIMT and branching', { interactive: true }),
          c('2.5', 'occupancy-latency-hiding', 'Occupancy and latency hiding', { interactive: true, lab: 'cpu' }),
          c('2.6', 'memory-wall-hbm', 'The memory wall and HBM'),
          c('2.7', 'designing-an-llm-processor', 'Thought experiment: designing an LLM processor'),
        ],
      },
      {
        id: 'm03',
        num: 3,
        title: 'The performance mental model',
        tagline: 'Roofline, arithmetic intensity, and why decode is memory-bound.',
        sources: [
          { id: 'BGE', note: 'Ch 33–35' },
          { id: 'TTFT', note: 'Week 1' },
          { id: 'MOHITH', note: 'Two phases of inference' },
          { id: 'FREGLY', note: 'Arithmetic intensity chapter' },
        ],
        chapters: [
          c('3.1', 'compute-vs-memory-bound', 'Compute-bound vs memory-bound'),
          c('3.2', 'arithmetic-intensity', 'Arithmetic intensity'),
          c('3.3', 'roofline', 'The roofline model', { interactive: true, lab: 'cpu' }),
          c('3.4', 'prefill-vs-decode', 'Prefill vs decode: TTFT and TPOT', { interactive: true }),
          c('3.5', 'derive-ai-for-7b', 'Lab: derive the arithmetic intensity of a 7B model', { lab: 'cpu' }),
        ],
      },
    ],
  },
  {
    num: 2,
    name: 'Intermediate',
    question: 'How do I write and optimize GPU code?',
    modules: [
      {
        id: 'm04',
        num: 4,
        title: 'CUDA programming fundamentals',
        tagline: 'Kernels, threads, blocks and memory: the CUDA programming model.',
        sources: [{ id: 'BGE', note: 'Phase 2, Ch 16–32' }, { id: 'CUDA100', note: 'Days 1–7' }],
        chapters: [
          c('4.1', 'kernels-are-not-functions', 'A kernel is not a function', { lab: 'gpu-required' }),
          c('4.2', 'thread-block-grid-indexing', 'Thread, block and grid indexing', { interactive: true, lab: 'gpu-required' }),
          c('4.3', 'host-device-memory', 'Host/device memory and transfers', { lab: 'gpu-required' }),
          c('4.4', 'shared-memory-syncthreads', 'Shared memory and __syncthreads', { lab: 'gpu-required' }),
          c('4.5', 'races-and-correctness', 'Race conditions and correctness', { lab: 'gpu-required' }),
          c('4.6', 'launch-async-streams', 'Kernel launch anatomy, async execution and streams', { interactive: true, lab: 'gpu-required' }),
          c('4.7', 'coalesced-vs-strided', 'Coalesced vs strided access', { interactive: true, lab: 'gpu-required' }),
          c('4.8', 'when-not-to-use-a-gpu', 'When not to use a GPU'),
        ],
      },
      {
        id: 'm05',
        num: 5,
        title: 'Parallel patterns',
        tagline: 'Reduction, scan, histogram and convolution: the building blocks of every kernel.',
        sources: [{ id: 'CUDA100', note: 'Days 8–25' }, { id: 'BGE', note: 'Ch 32, 47' }],
        chapters: [
          c('5.1', 'reduction-trees', 'Reduction trees', { interactive: true, lab: 'gpu-required' }),
          c('5.2', 'warp-shuffles', 'Warp shuffles', { lab: 'gpu-required' }),
          c('5.3', 'atomics-and-cas', 'Atomics and compare-and-swap', { lab: 'gpu-required' }),
          c('5.4', 'prefix-sum-scan', 'Prefix sum (scan)', { interactive: true, lab: 'gpu-required' }),
          c('5.5', 'histograms-privatization', 'Histograms and privatization', { lab: 'gpu-required' }),
          c('5.6', 'convolution-stencils', 'Convolution and stencils', { lab: 'gpu-required' }),
          c('5.7', 'softmax-kernels', 'Softmax: three kernels, then one fused kernel', { lab: 'gpu-required' }),
        ],
      },
      {
        id: 'm06',
        num: 6,
        title: 'GEMM and memory optimization',
        tagline: 'From a naive matmul to a tiled, tensor-core-aware kernel.',
        sources: [
          { id: 'BGE', note: 'Phase 3, Ch 36–51' },
          { id: 'CUDA100', note: 'Days 15, 33' },
          { id: 'FREGLY', note: 'Memory access, occupancy and ILP chapters' },
        ],
        chapters: [
          c('6.1', 'tiled-matmul', 'Naive → coalesced → shared-memory tiled matmul', { interactive: true, lab: 'gpu-required' }),
          c('6.2', 'register-tiling', 'Register tiling: the tile inside the tile', { lab: 'gpu-required' }),
          c('6.3', 'register-pressure-spilling', 'Register pressure, spilling and the occupancy trade-off'),
          c('6.4', 'bank-conflicts', 'Shared-memory bank conflicts', { interactive: true }),
          c('6.5', 'wave-quantization', 'Wave quantization', { interactive: true }),
          c('6.6', 'latency-hiding-ilp', 'Latency hiding and instruction-level parallelism'),
          c('6.7', 'tensor-cores-mixed-precision', 'Tensor cores and mixed precision', { lab: 'gpu-required' }),
          c('6.8', 'reading-ptx-sass', 'Reading PTX and SASS', { lab: 'gpu-required' }),
        ],
      },
      {
        id: 'm07',
        num: 7,
        title: "Profiling: measure, don't guess",
        tagline: 'Nsight, the PyTorch profiler, and a top-down profiling discipline.',
        sources: [{ id: 'BGE', note: 'Phase 4, Ch 52–56' }, { id: 'FREGLY', note: 'PyTorch profiling chapter' }],
        chapters: [
          c('7.1', 'top-down-profiling', 'The top-down profiling discipline'),
          c('7.2', 'nsight-systems-vs-compute', 'Nsight Systems vs Nsight Compute', { lab: 'gpu-required' }),
          c('7.3', 'what-kernel-time-measures', 'What kernel time actually measures'),
          c('7.4', 'pytorch-profiler', 'The PyTorch profiler and traces', { lab: 'gpu-optional' }),
          c('7.5', 'missing-tflops', 'Case study: the 91 missing TFLOPS', { lab: 'gpu-required' }),
        ],
      },
      {
        id: 'm08',
        num: 8,
        title: 'Transformer kernels and compilers',
        tagline: 'The kernels an LLM actually runs, plus Triton and torch.compile.',
        sources: [
          { id: 'CUDA100', note: 'Days 14, 20, 26–28, 31, 34, 38–41, 77' },
          { id: 'FREGLY', note: 'torch.compile / Triton chapter' },
        ],
        chapters: [
          c('8.1', 'norms-fused-residual', 'RMSNorm, LayerNorm and fused residual add', { lab: 'gpu-required' }),
          c('8.2', 'rope-alibi', 'RoPE and ALiBi', { interactive: true, lab: 'gpu-required' }),
          c('8.3', 'swiglu-mlp', 'The SwiGLU MLP block', { lab: 'gpu-required' }),
          c('8.4', 'mha-kernel', 'A multi-head attention kernel', { lab: 'gpu-required' }),
          c('8.5', 'flashattention', 'FlashAttention: tiling and online softmax', { interactive: true, lab: 'gpu-optional' }),
          c('8.6', 'triton', 'Triton for kernel authors', { lab: 'gpu-required' }),
          c('8.7', 'torch-compile-cuda-graphs', 'torch.compile and CUDA Graphs', { lab: 'gpu-required' }),
          c('8.8', 'fused-qkv-rope-kv', 'Fused QKV + RoPE + KV-cache update', { lab: 'gpu-required' }),
        ],
      },
    ],
  },
  {
    num: 3,
    name: 'Advanced',
    question: 'How do I build and tune a serving engine?',
    modules: [
      {
        id: 'm09',
        num: 9,
        title: 'KV cache engineering',
        tagline: 'The memory that decides how many users you can serve.',
        sources: [{ id: 'MOHITH', note: 'KV size, memory budget, capacity equation' }, { id: 'CUDA100', note: 'Days 35, 47, 74, 93, 100' }],
        chapters: [
          c('9.1', 'kv-cache-size', 'How big is the KV cache? The capacity equation', { interactive: true, lab: 'cpu' }),
          c('9.2', 'mqa-gqa-mla', 'MQA, GQA and MLA', { interactive: true }),
          c('9.3', 'paged-attention', 'PagedAttention and block tables', { interactive: true, lab: 'cpu' }),
          c('9.4', 'varlen-cu-seqlens', 'Variable-length batches (cu_seqlens)', { lab: 'gpu-optional' }),
          c('9.5', 'split-kv-flash-decoding', 'Split-KV and FlashDecoding', { interactive: true }),
          c('9.6', 'kv-cache-quantization', 'KV-cache quantization', { lab: 'gpu-optional' }),
          c('9.7', 'sinks-sliding-window-eviction', 'Sliding windows, attention sinks and eviction', { interactive: true }),
        ],
      },
      {
        id: 'm10',
        num: 10,
        title: 'Serving engine internals',
        tagline: 'Schedulers, batching and prefix caching inside vLLM and SGLang.',
        sources: [{ id: 'TTFT', note: 'Weeks 2 and 4' }, { id: 'MOHITH', note: 'Serving-level optimizations' }, { id: 'FREGLY', note: 'Dynamic scheduling chapter' }],
        chapters: [
          c('10.1', 'continuous-batching', 'Static vs continuous batching', { interactive: true }),
          c('10.2', 'scheduler', 'The scheduler: admission, preemption and swapping', { interactive: true }),
          c('10.3', 'chunked-prefill', 'Chunked prefill (Sarathi-Serve)', { interactive: true }),
          c('10.4', 'prefix-caching-radix', 'Prefix caching and RadixAttention', { interactive: true }),
          c('10.5', 'vllm-v1-architecture', 'A walkthrough of the vLLM V1 architecture'),
          c('10.6', 'sglang-vs-vllm', 'SGLang vs vLLM'),
          c('10.7', 'deploy-openai-compatible', 'Lab: deploy a 7–8B model behind an OpenAI-compatible API', { lab: 'gpu-required' }),
          c('10.8', 'mini-engine', 'Lab: build a mini engine (paged KV + continuous batching)', { lab: 'cpu' }),
        ],
      },
      {
        id: 'm11',
        num: 11,
        title: 'Decoding and sampling',
        tagline: 'How the next token is chosen, and how to choose several at once.',
        sources: [{ id: 'CUDA100', note: 'Days 68, 75, 79, 81' }, { id: 'TTFT', note: 'Week 7' }],
        chapters: [
          c('11.1', 'sampling-strategies', 'Greedy, top-k, top-p and min-p', { interactive: true, lab: 'cpu' }),
          c('11.2', 'beam-search', 'Beam search'),
          c('11.3', 'speculative-decoding', 'Speculative decoding: draft, verify, accept', { interactive: true }),
          c('11.4', 'eagle-medusa', 'EAGLE- and Medusa-style variants'),
          c('11.5', 'speculation-crossover', 'When speculation stops paying off: the QPS crossover', { lab: 'gpu-required' }),
        ],
      },
      {
        id: 'm12',
        num: 12,
        title: 'Quantization for inference',
        tagline: 'Fewer bits, fewer bytes moved, more tokens per second.',
        sources: [
          { id: 'TTFT', note: 'Week 6' },
          { id: 'CUDA100', note: 'Days 29, 51, 53' },
          { id: 'MOHITH', note: 'Weight quantization' },
          { id: 'FREGLY' },
        ],
        chapters: [
          c('12.1', 'number-formats', 'Number formats: FP32 → BF16 → FP8 → INT4', { interactive: true }),
          c('12.2', 'weight-only-vs-wa', 'Weight-only vs weight + activation quantization'),
          c('12.3', 'gptq-awq', 'GPTQ and AWQ'),
          c('12.4', 'w4a16-kernel', 'A W4A16 matmul kernel', { lab: 'gpu-required' }),
          c('12.5', 'fp8-serving', 'Serving in FP8', { lab: 'gpu-required' }),
          c('12.6', 'quantization-shootout', 'Lab: FP16 vs FP8 vs INT4 throughput and quality', { lab: 'gpu-required' }),
        ],
      },
      {
        id: 'm13',
        num: 13,
        title: 'Measurement and benchmarking',
        tagline: 'TTFT, ITL, goodput and tail latency, measured honestly.',
        sources: [{ id: 'TTFT', note: 'Weeks 3 and 5' }],
        chapters: [
          c('13.1', 'serving-metrics', 'Metrics: TTFT, ITL, TPOT, throughput and goodput', { interactive: true }),
          c('13.2', 'percentiles-tail-latency', 'Percentiles and tail latency', { interactive: true }),
          c('13.3', 'prometheus-grafana', 'Prometheus and Grafana for vLLM', { lab: 'gpu-required' }),
          c('13.4', 'load-testing', 'Load testing with GuideLLM and vllm bench serve', { lab: 'gpu-required' }),
          c('13.5', 'benchmark-pitfalls', 'Benchmarking pitfalls and reproducibility'),
        ],
      },
    ],
  },
  {
    num: 4,
    name: 'Expert',
    question: 'How do I run inference at scale?',
    modules: [
      {
        id: 'm14',
        num: 14,
        title: 'Multi-GPU inference and parallelism',
        tagline: 'Splitting a model across GPUs, and what that costs in communication.',
        sources: [{ id: 'FREGLY', note: 'Distributed networking and multinode chapters' }, { id: 'CUDA100', note: 'Days 67, 92' }],
        chapters: [
          c('14.1', 'collectives-nccl', 'Collectives and NCCL', { interactive: true }),
          c('14.2', 'tensor-parallelism', 'Tensor parallelism', { interactive: true }),
          c('14.3', 'pipeline-parallelism', 'Pipeline parallelism', { interactive: true }),
          c('14.4', 'data-parallel-replicas', 'Data parallelism and replicas'),
          c('14.5', 'moe-expert-parallelism', 'MoE: gating, grouped GEMM and expert parallelism', { interactive: true }),
          c('14.6', 'interconnects', 'Interconnects: NVLink, InfiniBand and RDMA'),
        ],
      },
      {
        id: 'm15',
        num: 15,
        title: 'Distributed serving and routing',
        tagline: 'Prefill/decode disaggregation, KV transfer, and smart routers.',
        sources: [
          { id: 'ROUTER' },
          { id: 'TTFT', note: 'Week 8' },
          { id: 'ICP' },
          { id: 'FREGLY', note: 'Disaggregated prefill/decode chapters' },
        ],
        chapters: [
          c('15.1', 'why-disaggregate', 'Why disaggregate prefill and decode', { interactive: true }),
          c('15.2', 'kv-transfer', 'KV transfer: NIXL and NCCL'),
          c('15.3', 'load-balancing-policies', 'Load balancing: round robin, power of two, KV affinity', { interactive: true }),
          c('15.4', 'vllm-router', 'Inside the vLLM Router'),
          c('15.5', 'resiliency', 'Resiliency: retries and circuit breakers'),
          c('15.6', 'kubernetes-autoscaling', 'Kubernetes, the production stack and queue-depth autoscaling', { lab: 'gpu-required' }),
        ],
      },
      {
        id: 'm16',
        num: 16,
        title: 'Economics and the inference control plane',
        tagline: 'Cost per token, SLO goodput, and the move from engine to control plane.',
        sources: [{ id: 'TTFT', note: 'Weeks 9–10' }, { id: 'ICP' }, { id: 'XPOST' }],
        chapters: [
          c('16.1', 'cost-per-million-tokens', 'Cost per million tokens', { interactive: true, lab: 'cpu' }),
          c('16.2', 'slos-goodput', 'SLOs and goodput'),
          c('16.3', 'cost-aware-routing', 'Cost-aware routing'),
          c('16.4', 'control-plane-llm-d', 'From engine to control plane: llm-d and the execution planner'),
          c('16.5', 'bottleneck-migration', 'Bottleneck migration'),
          c('16.6', 'reading-the-frontier', 'Reading the frontier: arXiv cs.DC, MLSys and OSDI'),
        ],
      },
      {
        id: 'capstone',
        num: 17,
        title: 'Capstone: ship one service',
        tagline: 'Deploy, instrument, load-test, tune, route, and publish a reproducible benchmark.',
        sources: [{ id: 'TTFT', note: 'The whole 10-week path' }],
        chapters: [
          c('C.1', 'deploy-and-instrument', 'Deploy and instrument', { lab: 'gpu-required' }),
          c('C.2', 'load-test-and-tune', 'Load-test, quantize and speculate', { lab: 'gpu-required' }),
          c('C.3', 'route-and-publish', 'Route and publish the benchmark', { lab: 'gpu-required' }),
        ],
      },
    ],
  },
];

export interface FlatChapter extends Chapter {
  module: Module;
  level: Level;
  path: string; // "/learn/m09/kv-cache-size"
  key: string; // "m09/kv-cache-size" (content collection id)
}

export const allModules: Module[] = levels.flatMap((l) => l.modules);

export const allChapters: FlatChapter[] = levels.flatMap((level) =>
  level.modules.flatMap((module) =>
    module.chapters.map((ch) => ({
      ...ch,
      module,
      level,
      key: `${module.id}/${ch.slug}`,
      path: `/learn/${module.id}/${ch.slug}`,
    })),
  ),
);

export const labMeta: Record<Lab, { label: string; short: string; color: string }> = {
  cpu: { label: 'Runs on a CPU or in the browser', short: 'CPU', color: 'var(--green)' },
  'gpu-optional': { label: 'GPU optional', short: 'GPU optional', color: 'var(--amber)' },
  'gpu-required': { label: 'GPU required (Colab or a rented GPU)', short: 'GPU', color: 'var(--red)' },
};
