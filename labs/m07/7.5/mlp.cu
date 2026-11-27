// Lab 7.5 — Case study: the missing TFLOPS
//
// The MLP block of a transformer layer, in FP16, for a prefill of 2048 tokens:
//
//     y = x + GELU(x W1) W2        x: 2048 x 2048, W1: 2048 x H, W2: H x 2048
//
// with H = 5461 (8/3 of 2048, the usual ratio for a gated MLP's hidden size). That is
// 2 * 2 * 2048 * 2048 * 5461 = 91.6 GFLOP of useful work per step.
//
// Version 0 is written the way such code often starts out. Each later version fixes one
// problem that profiling found, in the order the top-down method finds them:
//
//   v0  as written: copies x in and y out every step and checks y for NaNs on the CPU;
//       W2 was loaded in FP32, so the second GEMM runs as FP32 SGEMM; H = 5461;
//       GELU and the residual add are separate kernels
//   v1  keep the data on the GPU (no per-step copies or CPU check)            level 1: timeline
//   v2  W2 in FP16, so both GEMMs use tensor cores                           level 2: kernel table
//   v3  pad H to 5632 (a multiple of 256), with zero weights                 level 3: one kernel
//   v4  fuse GELU into GEMM 1 and the residual add into GEMM 2 (cuBLASLt)    level 2 again
//
// Every version computes the same y (padding adds only zeros) and is checked against v4.
//
// Build & run:  nvcc -O2 -arch=native -DUSE_CUBLAS mlp.cu -o mlp -lcublas -lcublasLt && ./mlp
//               ./mlp 0      runs only v0 (for profiling one version at a time)
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include "../common.cuh"

constexpr int M = 2048, D = 2048, H = 5461, HP = 5632;   // HP: H padded to a multiple of 256
constexpr double USEFUL_FLOP = 2.0 * 2 * M * D * double(H);

// GELU, tanh approximation (the one cuBLASLt's epilogue uses).
__device__ __forceinline__ float gelu(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}
// h (M x ld, half) -> GELU(h), as float (v0, v1: the FP32 GEMM needs FP32 input) or half.
__global__ void gelu_to_float(const half* h, float* out, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = gelu(__half2float(h[i]));
}
__global__ void gelu_half(half* h, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) h[i] = __float2half(gelu(__half2float(h[i])));
}
// y = x + t, with t in float (v0, v1) or half (v2, v3).
__global__ void residual_float(const half* x, const float* t, half* y, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __float2half(__half2float(x[i]) + t[i]);
}
__global__ void residual_half(const half* x, const half* t, half* y, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __hadd(x[i], t[i]);
}
__global__ void float_to_half(const float* a, half* b, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) b[i] = __float2half(a[i]);
}
// Copy a rows x cols matrix into a rows x ld one, zero-filling the extra columns.
template <class T>
__global__ void pad_cols(const T* src, T* dst, int rows, int cols, int ld) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i >= size_t(rows) * ld) return;
    int r = int(i / ld), c = int(i % ld);
    dst[i] = c < cols ? src[size_t(r) * cols + c] : T(0);
}

static int blocks(size_t n) { return int((n + 255) / 256); }

// Row-major C (m x n) = A (m x k) * B (k x n), FP16 in and out, FP32 accumulate, on tensor cores.
// cuBLAS is column-major, so ask for C^T = B^T A^T (see m06/common.cuh).
static void hgemm(cublasHandle_t h, int m, int n, int k, const half* A, int lda, const half* B, int ldb, half* C, int ldc) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, B, CUDA_R_16F, ldb, A, CUDA_R_16F, lda, &zero, C,
                              CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// cuBLASLt matmul with an optional fused epilogue: D = A B (+ C), then optionally GELU.
struct LtGemm {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
    // Row-major m x n = (m x k)(k x n); expressed column-major as n x m = (n x k)(k x m).
    LtGemm(cublasLtHandle_t lt, int m, int n, int k, cublasLtEpilogue_t epi, void* ws, size_t ws_bytes) {
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, n, k, n));   // "A" in column-major terms
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, k, m, k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, n, m, n));
        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
        cublasLtMatmulHeuristicResult_t res;
        int found = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, b, a, c, c, pref, 1, &res, &found));
        if (!found) { std::fprintf(stderr, "no cuBLASLt algorithm for this epilogue\n"); std::exit(1); }
        algo = res.algo;
        cublasLtMatmulPreferenceDestroy(pref);
    }
    // D = A B + beta * C
    void run(cublasLtHandle_t lt, const half* A, const half* B, const half* C, float beta, half* Dout, void* ws, size_t ws_bytes) {
        const float one = 1.0f;
        CUBLAS_CHECK(cublasLtMatmul(lt, op, &one, B, b, A, a, &beta, C, c, Dout, c, &algo, ws, ws_bytes, 0));
    }
};

struct Buffers {
    half *x, *y, *w1, *w1p, *w2h, *w2hp, *hid;   // hid: M x HP halves (GEMM 1 output)
    float *w2f, *act32, *t32;                     // FP32 copies for v0/v1
    half* t16;
    std::vector<half> host_x, host_y;             // pageable host buffers (v0)
};

static const char* names[] = {
    "v0 as written", "v1 data stays on GPU", "v2 W2 in FP16", "v3 H padded to 5632", "v4 GELU, residual fused"};

// One forward step of version v.
static void step(int v, Buffers& b, cublasHandle_t h, cublasLtHandle_t lt, LtGemm* g1, LtGemm* g2, void* ws, size_t wsb) {
    Range r(names[v]);
    const size_t nx = size_t(M) * D;
    if (v == 0) {
        Range rc("copy x in");
        CUDA_CHECK(cudaMemcpy(b.x, b.host_x.data(), nx * sizeof(half), cudaMemcpyHostToDevice));
    }
    const int ld = v >= 3 ? HP : H;              // hidden width actually used
    const size_t nh = size_t(M) * ld;
    if (v <= 1) {                                // GEMM 1 in FP16, then GELU to FP32, SGEMM, residual
        hgemm(h, M, H, D, b.x, D, b.w1, H, b.hid, H);
        gelu_to_float<<<blocks(nh), 256>>>(b.hid, b.act32, nh);
        sgemm_ref(h, M, D, H, b.act32, b.w2f, b.t32);
        residual_float<<<blocks(nx), 256>>>(b.x, b.t32, b.y, nx);
    } else if (v <= 3) {                         // both GEMMs FP16; GELU and residual still separate
        hgemm(h, M, ld, D, b.x, D, v == 3 ? b.w1p : b.w1, ld, b.hid, ld);
        gelu_half<<<blocks(nh), 256>>>(b.hid, nh);
        hgemm(h, M, D, ld, b.hid, ld, v == 3 ? b.w2hp : b.w2h, D, b.t16, D);
        residual_half<<<blocks(nx), 256>>>(b.x, b.t16, b.y, nx);
    } else {                                     // cuBLASLt: GELU in GEMM 1's epilogue, x as GEMM 2's C
        g1->run(lt, b.x, b.w1p, b.hid, 0.0f, b.hid, ws, wsb);
        g2->run(lt, b.hid, b.w2hp, b.x, 1.0f, b.y, ws, wsb);
    }
    CUDA_CHECK_LAUNCH();
    if (v == 0) {
        {
            Range rc("copy y out");
            CUDA_CHECK(cudaMemcpy(b.host_y.data(), b.y, nx * sizeof(half), cudaMemcpyDeviceToHost));
        }
        Range rn("NaN check on CPU");
        for (size_t i = 0; i < nx; i++)
            if (__hisnan(b.host_y[i])) { std::fprintf(stderr, "NaN at %zu\n", i); std::exit(1); }
    }
}

int main(int argc, char** argv) {
    int only = argc > 1 ? std::atoi(argv[1]) : -1;
    print_device();
    cublasHandle_t h;
    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasLtCreate(&lt));
    size_t wsb = 32 << 20;
    void* ws;
    CUDA_CHECK(cudaMalloc(&ws, wsb));
    CUBLAS_CHECK(cublasSetWorkspace(h, ws, wsb));

    // Weights scaled by 1/sqrt(fan-in) so activations stay O(1), as in a trained model.
    Buffers b;
    const size_t nx = size_t(M) * D;
    float* xf = random_matrix(nx, 1);
    float* w1f = random_matrix(size_t(D) * H, 2);
    b.w2f = random_matrix(size_t(H) * D, 3);
    {
        std::vector<float> t(size_t(D) * H);
        CUDA_CHECK(cudaMemcpy(t.data(), w1f, t.size() * 4, cudaMemcpyDeviceToHost));
        for (auto& v : t) v *= 1.0f / std::sqrt(float(D));
        CUDA_CHECK(cudaMemcpy(w1f, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(t.data(), b.w2f, t.size() * 4, cudaMemcpyDeviceToHost));
        for (auto& v : t) v = __half2float(__float2half(v / std::sqrt(float(H))));   // same values as the FP16 copy
        CUDA_CHECK(cudaMemcpy(b.w2f, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&b.x, nx * 2));
    CUDA_CHECK(cudaMalloc(&b.y, nx * 2));
    CUDA_CHECK(cudaMalloc(&b.t16, nx * 2));
    CUDA_CHECK(cudaMalloc(&b.t32, nx * 4));
    CUDA_CHECK(cudaMalloc(&b.w1, size_t(D) * H * 2));
    CUDA_CHECK(cudaMalloc(&b.w1p, size_t(D) * HP * 2));
    CUDA_CHECK(cudaMalloc(&b.w2h, size_t(H) * D * 2));
    CUDA_CHECK(cudaMalloc(&b.w2hp, size_t(HP) * D * 2));
    CUDA_CHECK(cudaMalloc(&b.hid, size_t(M) * HP * 2));
    CUDA_CHECK(cudaMalloc(&b.act32, size_t(M) * HP * 4));
    float_to_half<<<blocks(nx), 256>>>(xf, b.x, nx);
    float_to_half<<<blocks(size_t(D) * H), 256>>>(w1f, b.w1, size_t(D) * H);
    float_to_half<<<blocks(size_t(H) * D), 256>>>(b.w2f, b.w2h, size_t(H) * D);
    pad_cols<<<blocks(size_t(D) * HP), 256>>>(b.w1, b.w1p, D, H, HP);              // W1: extra zero columns
    CUDA_CHECK(cudaMemset(b.w2hp, 0, size_t(HP) * D * 2));                          // W2: extra zero rows
    CUDA_CHECK(cudaMemcpy(b.w2hp, b.w2h, size_t(H) * D * 2, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_LAUNCH();
    b.host_x.resize(nx);
    b.host_y.resize(nx);
    CUDA_CHECK(cudaMemcpy(b.host_x.data(), b.x, nx * 2, cudaMemcpyDeviceToHost));

    LtGemm g1(lt, M, HP, D, CUBLASLT_EPILOGUE_GELU, ws, wsb);
    LtGemm g2(lt, M, D, HP, CUBLASLT_EPILOGUE_DEFAULT, ws, wsb);

    // Reference output: v4.
    std::vector<half> ref(nx), out(nx);
    step(4, b, h, lt, &g1, &g2, ws, wsb);
    CUDA_CHECK(cudaMemcpy(ref.data(), b.y, nx * 2, cudaMemcpyDeviceToHost));

    std::printf("MLP block, FP16: x %d x %d, hidden %d; %.1f GFLOP of useful work per step\n", M, D, H, USEFUL_FLOP / 1e9);
    std::printf("  %-26s %10s %10s %14s\n", "version", "step", "TFLOP/s", "max |y - y4|");
    // The GPU's boost clock drifts as it heats up, so measuring v0, then v1, ... would
    // give later versions a different clock. Instead, interleave: in each round, time
    // every version for 10 steps; report each version's median over the rounds.
    const int rounds = only >= 0 ? 1 : 7, iters = 10;
    std::vector<double> ms[5];
    for (int v = 0; v < 5; v++)
        if (only < 0 || v == only)
            for (int i = 0; i < 3; i++) step(v, b, h, lt, &g1, &g2, ws, wsb);   // warm-up
    for (int r = 0; r < rounds; r++)
        for (int v = 0; v < 5; v++) {
            if (only >= 0 && v != only) continue;
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();                      // wall clock: the judge
            for (int i = 0; i < iters; i++) step(v, b, h, lt, &g1, &g2, ws, wsb);
            CUDA_CHECK(cudaDeviceSynchronize());
            ms[v].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters);
        }
    for (int v = 0; v < 5; v++) {
        if (only >= 0 && v != only) continue;
        step(v, b, h, lt, &g1, &g2, ws, wsb);
        CUDA_CHECK(cudaMemcpy(out.data(), b.y, nx * 2, cudaMemcpyDeviceToHost));
        float err = 0;
        for (size_t i = 0; i < nx; i++) err = std::fmax(err, std::fabs(__half2float(out[i]) - __half2float(ref[i])));
        std::sort(ms[v].begin(), ms[v].end());
        double med = ms[v][ms[v].size() / 2];
        std::printf("  %-26s %7.2f ms %10.2f %14.1e %s\n", names[v], med, USEFUL_FLOP / (med * 1e9), err,
                    err < 0.05f ? "PASS" : "FAIL");
    }

    if (only < 0) {
        // The ceiling: the same two GEMMs alone, at the padded shape, timed with events.
        float ms = time_ms([&] {
            hgemm(h, M, HP, D, b.x, D, b.w1p, HP, b.hid, HP);
            hgemm(h, M, D, HP, b.hid, HP, b.w2hp, D, b.t16, D);
        });
        std::printf("  %-26s %7.2f ms %10.2f   (both GEMMs alone, padded shape)\n", "ceiling: cuBLAS GEMMs", ms,
                    USEFUL_FLOP / (ms * 1e9));
    }
    std::printf("Done.\n");
    return 0;
}
