// Lab 7.2 — Nsight Systems vs Nsight Compute
//
// One small program to point both profilers at. Each step:
//
//   1. copies a 64 MB input from pinned host memory (async)
//   2. scale_coalesced:  y[i] = 2 x[i], neighbouring threads read neighbouring floats
//   3. scale_strided:    the same work, but neighbouring threads read floats 32 apart
//   4. sum_kernel:       adds up y into one float (block sums + one atomic per block)
//   5. reads the sum back with cudaMemcpy, which waits for everything above
//   6. "post-processes" on the CPU for 2 ms, while the GPU has nothing to do
//
// Nsight Systems shows where a step's time goes (copies, kernels, the CPU gap);
// Nsight Compute shows why scale_strided is slower than scale_coalesced.
//
// Build & run:  nvcc -O2 -arch=native -lineinfo tools.cu -o tools && ./tools
// Profile:      see README.md in this folder's parent, or the chapter.
#include <chrono>
#include "../common.cuh"

constexpr int N = 1 << 24;          // 16M floats, 64 MB
constexpr int STRIDE = 32;

__global__ void scale_coalesced(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}

// Thread i handles element (i % rows) * STRIDE + i / rows: every element exactly once,
// but the 32 lanes of a warp touch 32 different 128-byte lines.
__global__ void scale_strided(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = n / STRIDE;
    if (i < n) {
        int j = (i % rows) * STRIDE + i / rows;
        y[j] = 2.0f * x[j];
    }
}

__global__ void sum_kernel(const float* y, float* out, int n) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += y[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    __shared__ float warp_sums[32];
    if (threadIdx.x % 32 == 0) warp_sums[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? warp_sums[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) atomicAdd(out, s);
    }
}

static void cpu_postprocess(double ms) {           // stands in for detokenizing, logging, scheduling...
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < ms) {}
}

int main() {
    print_device();
    float *h_x, *x, *y, *sum;
    CUDA_CHECK(cudaMallocHost(&h_x, N * sizeof(float)));
    for (int i = 0; i < N; i++) h_x[i] = float(i % 7);            // sums to a known value
    CUDA_CHECK(cudaMalloc(&x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sum, sizeof(float)));
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    double expect = 0;
    for (int i = 0; i < N; i++) expect += 2.0 * h_x[i];

    const int steps = 10;
    float result = 0;
    double wall = 0;
    for (int it = 0; it < steps + 2; it++) {       // 2 warm-up steps
        auto t0 = std::chrono::steady_clock::now();
        {
            Range r("step");
            CUDA_CHECK(cudaMemcpyAsync(x, h_x, N * sizeof(float), cudaMemcpyHostToDevice, s));
            scale_coalesced<<<N / 256, 256, 0, s>>>(x, y, N);
            scale_strided<<<N / 256, 256, 0, s>>>(x, y, N);
            CUDA_CHECK(cudaMemsetAsync(sum, 0, sizeof(float), s));
            sum_kernel<<<160, 256, 0, s>>>(y, sum, N);
            CUDA_CHECK_LAUNCH();
            {
                Range rr("read sum back");
                CUDA_CHECK(cudaMemcpy(&result, sum, sizeof(float), cudaMemcpyDeviceToHost));   // waits for the stream
            }
            Range rc("CPU post-process");
            cpu_postprocess(2.0);
        }
        if (it >= 2) wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // Each kernel on its own, timed with events, for comparison with the profilers' numbers.
    float t_co = time_ms([&] { scale_coalesced<<<N / 256, 256>>>(x, y, N); });
    float t_st = time_ms([&] { scale_strided<<<N / 256, 256>>>(x, y, N); });
    const double gb = 2.0 * N * sizeof(float) / 1e9;   // read x, write y
    std::printf("step (wall clock, mean of %d): %.2f ms\n", steps, wall / steps);
    std::printf("scale_coalesced: %.3f ms  %6.1f GB/s\n", t_co, gb / (t_co * 1e-3));
    std::printf("scale_strided:   %.3f ms  %6.1f GB/s\n", t_st, gb / (t_st * 1e-3));
    std::printf("peak DRAM bandwidth (from clocks): %.1f GB/s\n", peak_bandwidth_gbs());
    bool ok = std::fabs(result - expect) <= 1e-3 * std::fabs(expect) + 1.0;
    std::printf("sum = %.0f (expected %.0f) %s\n", result, expect, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
