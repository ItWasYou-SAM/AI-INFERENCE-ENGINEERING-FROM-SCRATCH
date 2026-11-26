// Lab 5.2 — Warp shuffles
//
// Part 1 prints what one warp's registers hold after each step of
// __shfl_down_sync (tree reduction) and __shfl_xor_sync (butterfly), so you
// can see where the sum ends up in each.
//
// Part 2 sums 32 M floats with the grid-stride kernel from 5.1 in three
// flavours: the full shared-memory tree, shuffles for the last warp only,
// and shuffles everywhere (one shared-memory slot per warp). Then it sums
// the same data as 32 K rows of 1024, one block per row, where the tree
// runs far more often and its cost starts to show.
//
// Part 3 is the one-pass version: each block's total goes straight into a
// global atomicAdd, so the whole sum is one launch.
//
// Build & run:  nvcc -O2 -arch=native shuffles.cu -o shuffles && ./shuffles
#include <cmath>
#include <cstring>
#include <vector>
#include "../common.cuh"

constexpr int BLOCK = 256;
constexpr unsigned FULL = 0xffffffffu;   // all 32 lanes take part in the shuffle

// ---- Part 1: watch the registers ---------------------------------------------------------------
__global__ void show_shuffles(int* trace_down, int* trace_xor) {
    int lane = threadIdx.x;
    int a = lane, b = lane;                     // lane i starts with the value i; the sum is 496
    trace_down[lane] = a; trace_xor[lane] = b;
    int step = 1;
    for (int offset = 16; offset > 0; offset /= 2, step++) {
        a += __shfl_down_sync(FULL, a, offset);  // read a from lane + offset
        b += __shfl_xor_sync(FULL, b, offset);   // read b from lane ^ offset
        trace_down[step * 32 + lane] = a;
        trace_xor[step * 32 + lane] = b;
    }
}

// ---- Part 2: three block reductions ------------------------------------------------------------
__device__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(FULL, v, offset);
    return v;                                    // lane 0 holds the warp's sum
}

// Each thread's private partial sum. ROWS = false: the whole array, grid-stride
// (the 5.1 v5 loop). ROWS = true: block b sums row b, which has n elements.
template <bool ROWS>
__device__ float thread_partial(const float* in, int n) {
    if (ROWS) in += size_t(blockIdx.x) * n;
    int start = ROWS ? threadIdx.x : blockIdx.x * blockDim.x + threadIdx.x;
    int step = ROWS ? blockDim.x : gridDim.x * blockDim.x;
    float v = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    int n4 = n / 4;
    for (int i = start; i < n4; i += step) {
        float4 q = in4[i];
        v += (q.x + q.y) + (q.z + q.w);
    }
    for (int i = n4 * 4 + start; i < n; i += step) v += in[i];
    return v;
}

// A: the 5.1 v5 kernel — shared-memory tree all the way down (8 barriers).
template <bool ROWS>
__global__ void reduce_smem(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    s[tid] = thread_partial<ROWS>(in, n);
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

// B: shared-memory tree down to 32 values, then the first warp finishes with shuffles.
template <bool ROWS>
__global__ void reduce_smem_lastwarp(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    s[tid] = thread_partial<ROWS>(in, n);
    __syncthreads();
    for (int stride = BLOCK / 2; stride >= 32; stride /= 2) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {
        float v = warp_sum(s[tid]);
        if (tid == 0) out[blockIdx.x] = v;
    }
}

// C: every warp reduces its 32 values in registers; warp sums meet in shared
// memory (8 floats); warp 0 reduces those. One barrier in total.
template <bool ROWS>
__global__ void reduce_shfl(const float* in, float* out, int n) {
    __shared__ float warp_sums[BLOCK / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float v = warp_sum(thread_partial<ROWS>(in, n));
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < BLOCK / 32 ? warp_sums[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// ---- Part 2c: the reduction alone ----------------------------------------------------------------
// Each block does ITERS block-wide sums back to back, on values already in
// registers, so no memory traffic hides the cost. Each sum's result feeds the
// next, as in a kernel where the reduction is on the critical path.
constexpr int ITERS = 1000;

__global__ void bench_smem(float* out) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    float v = tid * 1e-3f;
    for (int it = 0; it < ITERS; it++) {
        s[tid] = v;
        __syncthreads();
        for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
            if (tid < stride) s[tid] += s[tid + stride];
            __syncthreads();
        }
        v = v * 0.5f + s[0] * 1e-6f;               // every thread reads the block's sum
        __syncthreads();                           // before s is overwritten
    }
    out[blockIdx.x * BLOCK + tid] = v;
}

__global__ void bench_shfl(float* out) {
    __shared__ float warp_sums[BLOCK / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float v = threadIdx.x * 1e-3f;
    for (int it = 0; it < ITERS; it++) {
        float w = warp_sum(v);
        if (lane == 0) warp_sums[warp] = w;
        __syncthreads();
        float total = warp_sum(lane < BLOCK / 32 ? warp_sums[lane] : 0.0f);   // every warp, redundantly
        total = __shfl_sync(FULL, total, 0);                                    // broadcast lane 0
        v = v * 0.5f + total * 1e-6f;
        __syncthreads();
    }
    out[blockIdx.x * BLOCK + threadIdx.x] = v;
}

// ---- Part 3: one launch, finishing with an atomic -------------------------------------------------
__global__ void reduce_shfl_atomic(const float* in, float* total, int n) {
    __shared__ float warp_sums[BLOCK / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float v = warp_sum(thread_partial<false>(in, n));
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = warp_sum(lane < BLOCK / 32 ? warp_sums[lane] : 0.0f);
        if (lane == 0) atomicAdd(total, v);      // one atomic per block, ~160 in total
    }
}

int main() {
    print_device();

    // Part 1
    int *d_down, *d_xor, h_down[6 * 32], h_xor[6 * 32];
    CUDA_CHECK(cudaMalloc(&d_down, sizeof h_down));
    CUDA_CHECK(cudaMalloc(&d_xor, sizeof h_xor));
    show_shuffles<<<1, 32>>>(d_down, d_xor);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_down, d_down, sizeof h_down, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_xor, d_xor, sizeof h_xor, cudaMemcpyDeviceToHost));
    auto show = [](const char* title, const int* t) {
        std::printf("\n%s (lanes 0-7 ... 31):\n", title);
        const char* names[] = {"start", "off 16", "off 8", "off 4", "off 2", "off 1"};
        for (int s = 0; s < 6; s++) {
            std::printf("  %-7s", names[s]);
            for (int l = 0; l < 8; l++) std::printf(" %4d", t[s * 32 + l]);
            std::printf("  ... %4d\n", t[s * 32 + 31]);
        }
    };
    show("__shfl_down_sync: the sum (496) ends up in lane 0", h_down);
    show("__shfl_xor_sync: every lane ends up with the sum", h_xor);

    // Part 2
    const int n = 1 << 25;
    std::vector<float> h(n);
    double expect = 0;
    for (int i = 0; i < n; i++) { h[i] = float(i % 17) * 0.125f; expect += h[i]; }
    float *in, *partial, *result;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&partial, 4096 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = sms * 8;

    std::printf("\nSum of %d floats, %d blocks of %d threads:\n", n, blocks, BLOCK);
    auto report = [&](const char* name, float ms) {
        float got;
        CUDA_CHECK(cudaMemcpy(&got, result, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = std::fabs(got - expect) <= 1e-5 * expect;
        std::printf("  %-38s %6.3f ms %7.1f GB/s  %s\n", name, ms, n * 4 / ms / 1e6, ok ? "PASS" : "FAIL");
    };
    struct K { const char* name; void (*whole)(const float*, float*, int); void (*rows)(const float*, float*, int); };
    K ks[] = {{"A shared-memory tree (8 barriers)", reduce_smem<false>, reduce_smem<true>},
              {"B tree to 32, then one warp shuffles", reduce_smem_lastwarp<false>, reduce_smem_lastwarp<true>},
              {"C shuffles everywhere (1 barrier)", reduce_shfl<false>, reduce_shfl<true>}};
    for (auto& k : ks) {
        float ms = time_ms([&] {
            k.whole<<<blocks, BLOCK>>>(in, partial, n);
            k.whole<<<1, BLOCK>>>(partial, result, blocks);
        });
        CUDA_CHECK_LAUNCH();
        report(k.name, ms);
    }

    // Part 2b: the same array as 32768 rows of 1024, one block per row.
    // Each thread now loads one float4 and the tree runs once per 4 KB, not once per 800 KB.
    const int cols = 1024, rows = n / cols;
    float* row_sums;
    CUDA_CHECK(cudaMalloc(&row_sums, rows * sizeof(float)));
    std::vector<float> got_rows(rows);
    std::printf("\nRow sums: %d rows of %d floats, one block per row:\n", rows, cols);
    for (auto& k : ks) {
        float ms = time_ms([&] { k.rows<<<rows, BLOCK>>>(in, row_sums, cols); });
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(got_rows.data(), row_sums, rows * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < rows; r++) {
            double e = 0;
            for (int c = 0; c < cols; c++) e += h[size_t(r) * cols + c];
            bad += std::fabs(got_rows[r] - e) > 1e-3;
        }
        std::printf("  %-38s %6.3f ms %7.1f GB/s  %s\n", k.name, ms, n * 4 / ms / 1e6, bad ? "FAIL" : "PASS");
    }
    cudaFree(row_sums);

    // Part 2c
    float* sink;
    CUDA_CHECK(cudaMalloc(&sink, sms * BLOCK * sizeof(float)));
    std::printf("\nThe block reduction alone (one block per SM, %d back-to-back sums of %d values):\n", ITERS, BLOCK);
    float t_smem = time_ms([&] { bench_smem<<<sms, BLOCK>>>(sink); }, 5);
    float t_shfl = time_ms([&] { bench_shfl<<<sms, BLOCK>>>(sink); }, 5);
    CUDA_CHECK_LAUNCH();
    std::printf("  %-38s %7.1f ns per block sum\n", "shared-memory tree (10 barriers)", t_smem * 1e6 / ITERS);
    std::printf("  %-38s %7.1f ns per block sum\n", "shuffles (2 barriers)", t_shfl * 1e6 / ITERS);
    cudaFree(sink);

    // Part 3
    std::printf("\nWhole array in one launch:\n");
    float ms = time_ms([&] {
        cudaMemsetAsync(result, 0, sizeof(float));
        reduce_shfl_atomic<<<blocks, BLOCK>>>(in, result, n);
    });
    CUDA_CHECK_LAUNCH();
    report("D shuffles + one atomicAdd per block", ms);

    // The atomic version adds the block totals in whatever order blocks finish.
    std::printf("\nOne-pass sum, 5 runs (bit patterns):");
    for (int r = 0; r < 5; r++) {
        CUDA_CHECK(cudaMemset(result, 0, sizeof(float)));
        reduce_shfl_atomic<<<blocks, BLOCK>>>(in, result, n);
        float got;
        CUDA_CHECK(cudaMemcpy(&got, result, sizeof(float), cudaMemcpyDeviceToHost));
        unsigned bits;
        std::memcpy(&bits, &got, 4);
        std::printf(" %08x", bits);
    }
    std::printf("\n");

    cudaFree(d_down); cudaFree(d_xor); cudaFree(in); cudaFree(partial); cudaFree(result);
}
