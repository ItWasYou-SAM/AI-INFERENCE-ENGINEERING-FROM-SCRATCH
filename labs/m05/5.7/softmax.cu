// Lab 5.7 — Softmax: three kernels, then one fused kernel
//
// Row-wise softmax y = exp(x - max) / sum(exp(x - max)) on three shapes:
//   attention   16384 rows x 1024    (scores of 16 heads x 1024 queries over 1024 keys)
//   vocab          64 rows x 128256  (logits for a decode batch of 64, Llama-3 vocabulary)
//   short      262144 rows x 64      (many tiny rows)
//
// Kernels, from most memory traffic to least:
//   3 kernels    max kernel, exp+sum kernel, normalize kernel      reads 3 rows, writes 2
//   fused 3-pass one block per row: max pass, sum pass, write pass  reads 3, writes 1
//   online       one pass for max and sum together, then write      reads 2, writes 1
//   registers    the row is loaded once into registers              reads 1, writes 1
//   warp/row     like registers, one warp per row, no shared memory reads 1, writes 1
//
// "GB/s" counts only the minimum traffic (read the row once, write it once),
// so it says how close each kernel gets to a perfect one.
//
// Build & run:  nvcc -O2 -arch=native softmax.cu -o softmax && ./softmax
#include <cfloat>
#include <cmath>
#include <random>
#include <vector>
#include "../common.cuh"

constexpr int BLOCK = 256;
constexpr unsigned FULL = 0xffffffffu;

// ---- block-wide reductions; every thread gets the result ----------------------------------------
__device__ float warp_max(float v) { for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(FULL, v, o)); return v; }
__device__ float warp_sum(float v) { for (int o = 16; o > 0; o /= 2) v += __shfl_xor_sync(FULL, v, o); return v; }

template <bool MAX>
__device__ float block_reduce(float v) {
    __shared__ float tmp[BLOCK / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();                            // tmp may still be in use by a previous call
    if (lane == 0) tmp[warp] = v;
    __syncthreads();
    v = lane < BLOCK / 32 ? tmp[lane] : (MAX ? -FLT_MAX : 0.0f);
    return MAX ? warp_max(v) : warp_sum(v);
}

// ---- 1. three kernels ------------------------------------------------------------------------------
__global__ void k_row_max(const float* x, float* m, int cols) {
    const float* row = x + size_t(blockIdx.x) * cols;
    float v = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += BLOCK) v = fmaxf(v, row[c]);
    v = block_reduce<true>(v);
    if (threadIdx.x == 0) m[blockIdx.x] = v;
}
__global__ void k_exp_sum(const float* x, const float* m, float* y, float* s, int cols) {
    size_t off = size_t(blockIdx.x) * cols;
    float mx = m[blockIdx.x], v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) { float e = expf(x[off + c] - mx); y[off + c] = e; v += e; }
    v = block_reduce<false>(v);
    if (threadIdx.x == 0) s[blockIdx.x] = v;
}
__global__ void k_normalize(float* y, const float* s, int cols) {
    size_t off = size_t(blockIdx.x) * cols;
    float inv = 1.0f / s[blockIdx.x];
    for (int c = threadIdx.x; c < cols; c += BLOCK) y[off + c] *= inv;
}

// ---- 2. fused, three passes over the row ----------------------------------------------------------
template <bool SAFE>
__global__ void k_fused3(const float* x, float* y, int cols) {
    const float* row = x + size_t(blockIdx.x) * cols;
    float* out = y + size_t(blockIdx.x) * cols;
    float mx = 0.0f;
    if (SAFE) {                                 // SAFE = false skips the max: see the stability demo
        float v = -FLT_MAX;
        for (int c = threadIdx.x; c < cols; c += BLOCK) v = fmaxf(v, row[c]);
        mx = block_reduce<true>(v);
    }
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) s += expf(row[c] - mx);
    float inv = 1.0f / block_reduce<false>(s);
    for (int c = threadIdx.x; c < cols; c += BLOCK) out[c] = expf(row[c] - mx) * inv;
}

// ---- 3. online softmax: max and sum in one pass ------------------------------------------------------
// Keep a running max m and a running sum d of exp(x - m). When a bigger value
// arrives, rescale d to the new max: d * exp(m_old - m_new).
__device__ void online_merge(float& m, float& d, float m2, float d2) {
    float mn = fmaxf(m, m2);
    d = d * expf(m - mn) + d2 * expf(m2 - mn);
    m = mn;
}
__global__ void k_online(const float* x, float* y, int cols) {
    const float* row = x + size_t(blockIdx.x) * cols;
    float* out = y + size_t(blockIdx.x) * cols;
    float m = -FLT_MAX, d = 0.0f;
    for (int c = threadIdx.x; c < cols; c += BLOCK) {
        float v = row[c];
        if (v > m) { d = d * expf(m - v) + 1.0f; m = v; }
        else d += expf(v - m);
    }
    for (int o = 16; o > 0; o /= 2) online_merge(m, d, __shfl_xor_sync(FULL, m, o), __shfl_xor_sync(FULL, d, o));
    __shared__ float sm[BLOCK / 32], sd[BLOCK / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) { sm[warp] = m; sd[warp] = d; }
    __syncthreads();
    m = lane < BLOCK / 32 ? sm[lane] : -FLT_MAX;
    d = lane < BLOCK / 32 ? sd[lane] : 0.0f;
    for (int o = 16; o > 0; o /= 2) online_merge(m, d, __shfl_xor_sync(FULL, m, o), __shfl_xor_sync(FULL, d, o));
    float inv = 1.0f / d;
    for (int c = threadIdx.x; c < cols; c += BLOCK) out[c] = expf(row[c] - m) * inv;
}

// ---- 4. the row lives in registers: one read, one write ---------------------------------------------
template <int COLS>
__global__ void k_registers(const float* x, float* y) {
    constexpr int PER = COLS / BLOCK;           // values per thread
    const float* row = x + size_t(blockIdx.x) * COLS;
    float v[PER];
    float mx = -FLT_MAX;
    #pragma unroll
    for (int j = 0; j < PER; j++) { v[j] = row[j * BLOCK + threadIdx.x]; mx = fmaxf(mx, v[j]); }
    mx = block_reduce<true>(mx);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < PER; j++) { v[j] = expf(v[j] - mx); s += v[j]; }
    float inv = 1.0f / block_reduce<false>(s);
    #pragma unroll
    for (int j = 0; j < PER; j++) y[size_t(blockIdx.x) * COLS + j * BLOCK + threadIdx.x] = v[j] * inv;
}

// ---- 5. one warp per row: only shuffles, no shared memory, no barriers --------------------------------
template <int COLS>
__global__ void k_warp_row(const float* x, float* y, int rows) {
    constexpr int PER = COLS / 32;
    int row = blockIdx.x * (BLOCK / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* in = x + size_t(row) * COLS;
    float v[PER];
    float mx = -FLT_MAX;
    #pragma unroll
    for (int j = 0; j < PER; j++) { v[j] = in[j * 32 + lane]; mx = fmaxf(mx, v[j]); }
    mx = warp_max(mx);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < PER; j++) { v[j] = expf(v[j] - mx); s += v[j]; }
    float inv = 1.0f / warp_sum(s);
    #pragma unroll
    for (int j = 0; j < PER; j++) y[size_t(row) * COLS + j * 32 + lane] = v[j] * inv;
}

// ---- driver -----------------------------------------------------------------------------------------
template <int COLS>
void bench(const char* label, int rows, std::mt19937& rng) {
    const size_t n = size_t(rows) * COLS;
    std::vector<float> h(n), got(n);
    std::normal_distribution<float> N(0.0f, 4.0f);
    for (auto& v : h) v = N(rng);
    float *x, *y, *m, *s;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto check = [&] {                          // compare 16 rows with a double-precision reference
        CUDA_CHECK(cudaMemcpy(got.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = 0; k < 16; k++) {
            int r = (k * 7919) % rows;
            const float* row = h.data() + size_t(r) * COLS;
            double mx = -1e30, sum = 0;
            for (int c = 0; c < COLS; c++) mx = std::max(mx, double(row[c]));
            for (int c = 0; c < COLS; c++) sum += std::exp(row[c] - mx);
            for (int c = 0; c < COLS; c++) {
                double e = std::exp(row[c] - mx) / sum;
                if (std::fabs(got[size_t(r) * COLS + c] - e) > 1e-6 + 1e-4 * e) return false;
            }
        }
        return true;
    };
    std::printf("\n%s: %d rows x %d (%.0f MB)\n", label, rows, COLS, n * 4 / 1e6);
    auto report = [&](const char* name, float ms) {
        bool ok = check();
        std::printf("  %-14s %8.3f ms %7.1f GB/s  %s\n", name, ms, 2.0 * n * 4 / ms / 1e6, ok ? "PASS" : "FAIL");
    };
    report("3 kernels", time_ms([&] {
        k_row_max<<<rows, BLOCK>>>(x, m, COLS);
        k_exp_sum<<<rows, BLOCK>>>(x, m, y, s, COLS);
        k_normalize<<<rows, BLOCK>>>(y, s, COLS);
    }, 10));
    report("fused 3-pass", time_ms([&] { k_fused3<true><<<rows, BLOCK>>>(x, y, COLS); }, 10));
    report("online", time_ms([&] { k_online<<<rows, BLOCK>>>(x, y, COLS); }, 10));
    if constexpr (COLS % BLOCK == 0 && COLS / BLOCK <= 16)
        report("registers", time_ms([&] { k_registers<COLS><<<rows, BLOCK>>>(x, y); }, 10));
    if constexpr (COLS % 32 == 0 && COLS / 32 <= 32)
        report("warp/row", time_ms([&] { k_warp_row<COLS><<<(rows + 7) / 8, BLOCK>>>(x, y, rows); }, 10));
    CUDA_CHECK_LAUNCH();
    cudaFree(x); cudaFree(y); cudaFree(m); cudaFree(s);
}

int main() {
    print_device();
    std::mt19937 rng(11);

    // Why subtract the max: logits of 90..100 overflow exp() in FP32 (max ~3.4e38 = e^88.7).
    {
        const int cols = 8;
        float h[cols] = {90, 92, 94, 96, 97, 98, 99, 100}, out[cols];
        float *x, *y;
        CUDA_CHECK(cudaMalloc(&x, sizeof h));
        CUDA_CHECK(cudaMalloc(&y, sizeof h));
        CUDA_CHECK(cudaMemcpy(x, h, sizeof h, cudaMemcpyHostToDevice));
        std::printf("\nLogits 90 ... 100:\n");
        for (int safe = 0; safe < 2; safe++) {
            if (safe) k_fused3<true><<<1, BLOCK>>>(x, y, cols); else k_fused3<false><<<1, BLOCK>>>(x, y, cols);
            CUDA_CHECK(cudaMemcpy(out, y, sizeof out, cudaMemcpyDeviceToHost));
            std::printf("  %-22s", safe ? "exp(x - max) / sum" : "exp(x) / sum");
            for (float v : out) std::printf(" %7.4f", v);
            std::printf("\n");
        }
        cudaFree(x); cudaFree(y);
    }

    bench<1024>("attention", 16384, rng);
    bench<128256>("vocab", 64, rng);
    bench<64>("short", 262144, rng);
}
