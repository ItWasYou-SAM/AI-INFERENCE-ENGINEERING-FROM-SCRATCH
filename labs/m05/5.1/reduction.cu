// Lab 5.1 — Reduction trees
//
// Sums 32 M floats five ways, each a small change from the last, and reports
// time and effective bandwidth. A sum reads every byte once and does one add
// per element, so it is memory-bound: the target is the GPU's DRAM bandwidth.
//
//   v1  interleaved addressing, divergent:   if (tid % (2*s) == 0)
//   v2  interleaved addressing, compact:     index = 2*s*tid
//   v3  sequential addressing:               if (tid < s) s[tid] += s[tid + s]
//   v4  add two elements during the load:    half the blocks
//   v5  grid-stride loop, float4 loads:      a few blocks per SM, many elements per thread
//
// Every version writes one partial sum per block; a second launch of the same
// kernel sums the partials (a "two-pass" reduction).
//
// Build & run:  nvcc -O2 -arch=native reduction.cu -o reduction && ./reduction
#include <cmath>
#include <vector>
#include "../common.cuh"

constexpr int BLOCK = 256;

// v1: thread tid works at step s only if tid is a multiple of 2s.
// Active threads are scattered across every warp, so all warps stay busy
// (and diverge) until the very last steps.
__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x, i = blockIdx.x * BLOCK + tid;
    s[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < BLOCK; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// v2: same tree, but thread tid handles pair number tid, so the active threads
// are packed into the first warps. No divergence until fewer than 32 remain,
// but the strided shared-memory index causes bank conflicts (6.4).
__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x, i = blockIdx.x * BLOCK + tid;
    s[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = 1; stride < BLOCK; stride *= 2) {
        int idx = 2 * stride * tid;
        if (idx < BLOCK) s[idx] += s[idx + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// v3: sequential addressing. Fold the top half onto the bottom half.
// Active threads are contiguous and so are their addresses: no divergence
// and no bank conflicts. This is the kernel from 4.4.
__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x, i = blockIdx.x * BLOCK + tid;
    s[tid] = i < n ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// v4: in v3, half the threads are idle from the first step. Give each thread
// two elements and add them while loading: half as many blocks, same tree.
__global__ void reduce_v4(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x, i = blockIdx.x * (2 * BLOCK) + tid;
    float v = i < n ? in[i] : 0.0f;
    if (i + BLOCK < n) v += in[i + BLOCK];
    s[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// v5: launch only enough blocks to fill the GPU. Each thread walks the array
// with a grid-stride loop, reading 16 bytes at a time, and keeps a running
// sum in a register. The shared-memory tree runs once per block, at the end.
__global__ void reduce_v5(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    float v = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    int n4 = n / 4;
    for (int i = blockIdx.x * BLOCK + tid; i < n4; i += gridDim.x * BLOCK) {
        float4 q = in4[i];
        v += (q.x + q.y) + (q.z + q.w);
    }
    for (int i = n4 * 4 + blockIdx.x * BLOCK + tid; i < n; i += gridDim.x * BLOCK) v += in[i];  // tail
    s[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

int main() {
    print_device();
    const int n = 1 << 25;                       // 32 M floats = 128 MB
    std::vector<float> h(n);
    double expect = 0;
    for (int i = 0; i < n; i++) { h[i] = float(i % 17) * 0.125f; expect += h[i]; }

    float *in, *partial, *result;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&partial, (n / BLOCK + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int v5_blocks = sms * 8;               // ~8 blocks of 256 threads per SM

    struct Version { const char* name; void (*k)(const float*, float*, int); int per_block; };
    Version versions[] = {
        {"v1 interleaved, divergent", reduce_v1, BLOCK},
        {"v2 interleaved, compact", reduce_v2, BLOCK},
        {"v3 sequential addressing", reduce_v3, BLOCK},
        {"v4 add during load", reduce_v4, 2 * BLOCK},
        {"v5 grid-stride + float4", reduce_v5, 0},
    };

    std::printf("\nSum of %d floats (%.0f MB), two passes:\n", n, n * 4 / 1e6);
    std::printf("  %-28s %9s %9s %14s\n", "kernel", "time", "GB/s", "result");
    for (auto& v : versions) {
        int blocks = v.per_block ? (n + v.per_block - 1) / v.per_block : v5_blocks;
        auto run = [&] {
            v.k<<<blocks, BLOCK>>>(in, partial, n);              // pass 1: one partial per block
            // pass 2: sum the partials with the same kernel (v5 handles any size in one block)
            reduce_v5<<<1, BLOCK>>>(partial, result, blocks);
        };
        float ms = time_ms(run);
        CUDA_CHECK_LAUNCH();
        float got;
        CUDA_CHECK(cudaMemcpy(&got, result, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = std::fabs(got - expect) <= 1e-5 * std::fabs(expect) + 1.0;
        std::printf("  %-28s %6.3f ms %9.1f %14.0f %s\n", v.name, ms, n * 4 / ms / 1e6, got, ok ? "PASS" : "FAIL");
    }
    std::printf("  %-28s %9s %9s %14.0f\n", "CPU (double)", "", "", expect);

    // For reference: a plain device-to-device copy reads and writes every byte once.
    float* tmp;
    CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(float)));
    float ms = time_ms([&] { cudaMemcpy(tmp, in, n * sizeof(float), cudaMemcpyDeviceToDevice); });
    std::printf("\n  cudaMemcpy D2D of the same array: %.1f GB/s (read + write)\n", 2.0 * n * 4 / ms / 1e6);

    cudaFree(in); cudaFree(partial); cudaFree(result); cudaFree(tmp);
}
