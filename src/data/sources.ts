export type SourceId = 'BGE' | 'TTFT' | 'CUDA100' | 'ROUTER' | 'MOHITH' | 'XPOST' | 'ICP' | 'FREGLY';

export interface Source {
  id: SourceId;
  title: string;
  author: string;
  kind: 'repo' | 'course' | 'blog' | 'paper' | 'book' | 'post';
  url?: string;
  description: string;
  placeholder?: boolean;
}

export const sources: Record<SourceId, Source> = {
  BGE: {
    id: 'BGE',
    title: 'BEFORE-GPU-ENGINEERING',
    author: 'BEFORE-GPU-ENGINEERING',
    kind: 'repo',
    url: 'https://github.com/BEFORE-GPU-ENGINEERING/BEFORE-GPU-ENGINEERING',
    description:
      'First-principles notebook on GPU architecture, published as 56 chapters in 4 phases. It covers warps, SIMT, the memory hierarchy, occupancy, coalescing, tiling and profiling, before any CUDA syntax.',
  },
  TTFT: {
    id: 'TTFT',
    title: 'time-to-first-token',
    author: 'patchy631',
    kind: 'course',
    url: 'https://github.com/patchy631/time-to-first-token',
    description:
      'A 10-week, 30-minutes-a-day roadmap. You ship one OpenAI-compatible vLLM service, instrument it, load-test it, quantize it, add speculative decoding, route it, and publish a benchmark.',
  },
  CUDA100: {
    id: 'CUDA100',
    title: '100 Days of CUDA',
    author: 'Pavel Simo',
    kind: 'course',
    url: 'https://pavelsimo.github.io/100-days-of-cuda/',
    description:
      'A daily CUDA log that runs from vector add to LLM kernels: reductions, scans, tiled GEMM, attention variants, RoPE, quantized matmul, sampling and paged KV-cache attention.',
  },
  ROUTER: {
    id: 'ROUTER',
    title: 'vLLM Router release',
    author: 'vLLM Team',
    kind: 'blog',
    url: 'https://vllm.ai/blog/2025-12-13-vllm-router-release',
    description:
      'A Rust load balancer for vLLM fleets. Covers consistent hashing for KV-cache affinity, power-of-two choices, prefill/decode disaggregation (NIXL/NCCL), retries and circuit breakers.',
  },
  MOHITH: {
    id: 'MOHITH',
    title: 'LLM Inference at Scale',
    author: 'Mohith',
    kind: 'blog',
    url: 'https://www.mohith.xyz/blog/llm-inference-at-scale',
    description:
      'Explains inference from the ground up: KV-cache size, the GPU memory budget, prefill vs decode, the capacity equation, GQA/MLA/FlashAttention, PagedAttention, continuous batching and speculative decoding.',
  },
  XPOST: {
    id: 'XPOST',
    title: 'Akshay Pachaar on X',
    author: 'Akshay Pachaar',
    kind: 'post',
    url: 'https://x.com/akshay_pachaar/status/2020840784782913605',
    description: 'Placeholder until the post content is added.',
    placeholder: true,
  },
  ICP: {
    id: 'ICP',
    title: 'From Inference Engine to Inference Control Plane',
    author: 'Twinkll Sisodia',
    kind: 'paper',
    url: 'https://arxiv.org/abs/2609.23130v1',
    description:
      'Treats vLLM as the execution layer and llm-d as the control layer. Proposes an inference execution planner, a bottleneck-migration taxonomy and SLO-goodput evaluation.',
  },
  FREGLY: {
    id: 'FREGLY',
    title: 'AI Systems Performance Engineering',
    author: 'Chris Fregly',
    kind: 'book',
    description:
      'Optimizing training and inference workloads with GPUs, CUDA and PyTorch: GPU architecture, memory access, occupancy, pipelining, PyTorch profiling, torch.compile/Triton, and disaggregated multinode inference.',
  },
};
