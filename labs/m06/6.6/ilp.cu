// Lab 6.6 — Latency hiding and instruction-level parallelism
//
// A GPU hides latency by always having another instruction ready to issue. That
// instruction can come from another warp (occupancy, 2.5) or from the same warp,
// if it doesn't depend on the one before (instruction-level parallelism, ILP).
//
// Part 1, arithmetic: each thread runs ILP independent chains of dependent FMAs.
//   With 1 chain, every FMA waits ~4 cycles for the previous one.
// Part 2, memory: each thread keeps ILP independent loads in flight per iteration.
//   Bandwidth = bytes in flight / latency (Little's law).
// Both sweep the number of warps per SM, to show that ILP and occupancy trade off.
//
// Build & run:  nvcc -O2 -arch=native ilp.cu -o ilp && ./ilp
#include "../common.cuh"

template <int ILP>
__global__ void fma_chains(float* out, int iters) {
    float a[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) a[j] = threadIdx.x * 1e-3f + j;
    const float b = 0.9999f, c = 1e-4f;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = fmaf(a[j], b, c);   // chain j depends only on itself
    }
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < ILP; j++) s += a[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int ILP>
__global__ void loads_in_flight(const float* __restrict__ x, float* out, size_t n) {
    float acc = 0.0f;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (ILP - 1) * stride < n; i += ILP * stride) {
        float v[ILP];
        #pragma unroll
        for (int j = 0; j < ILP; j++) v[j] = x[i + j * stride];   // ILP independent loads, issued back to back
        #pragma unroll
        for (int j = 0; j < ILP; j++) acc += v[j];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main() {
    print_device();
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    float* out;
    CUDA_CHECK(cudaMalloc(&out, size_t(sms) * 2048 * sizeof(float)));

    // Part 1: one block per SM, with 4 to 32 warps
    const int iters = 1 << 14;
    std::printf("\nPart 1: FMA throughput, GFLOP/s (one block per SM; columns = independent chains per thread)\n");
    std::printf("  %14s %9s %9s %9s %9s\n", "warps per SM", "ILP 1", "ILP 2", "ILP 4", "ILP 8");
    for (int warps : {4, 8, 16, 32}) {
        int threads = warps * 32;
        auto run = [&](auto k, int ilp) {
            float ms = time_ms([&] { k<<<sms, threads>>>(out, iters / ilp); }, 5);
            return 2.0 * sms * threads * double(iters) / ms / 1e6;   // same FMA count for every ILP
        };
        double g1 = run(fma_chains<1>, 1), g2 = run(fma_chains<2>, 2), g4 = run(fma_chains<4>, 4), g8 = run(fma_chains<8>, 8);
        CUDA_CHECK_LAUNCH();
        std::printf("  %5d (%2d/sched) %9.0f %9.0f %9.0f %9.0f\n", warps, warps / 4, g1, g2, g4, g8);
    }

    // Part 2: two blocks per SM, with 2 to 32 warps each
    const size_t n = size_t(1) << 26;            // 256 MB
    float* x = random_matrix(n, 3);
    std::printf("\nPart 2: read bandwidth, GB/s (two blocks per SM; columns = independent loads in flight per thread)\n");
    std::printf("  %14s %9s %9s %9s %9s\n", "warps per SM", "ILP 1", "ILP 2", "ILP 4", "ILP 8");
    for (int warps : {4, 8, 16, 32, 64}) {
        int threads = warps * 16;                // two blocks per SM
        auto run = [&](auto k) {
            float ms = time_ms([&] { k<<<2 * sms, threads>>>(x, out, n); }, 5);
            return n * 4.0 / ms / 1e6;
        };
        double b1 = run(loads_in_flight<1>), b2 = run(loads_in_flight<2>), b4 = run(loads_in_flight<4>), b8 = run(loads_in_flight<8>);
        CUDA_CHECK_LAUNCH();
        std::printf("  %14d %9.1f %9.1f %9.1f %9.1f\n", warps, b1, b2, b4, b8);
    }
    cudaFree(out); cudaFree(x);
}
