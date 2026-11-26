// Lab 6.7 — Tensor cores and mixed precision
//
// The same 4096 x 4096 x 4096 matmul on CUDA cores and on tensor cores, in
// different precisions, with the error of each against an FP32 result.
//
//   cuBLAS FP32         SGEMM on the ordinary FP32 cores (the reference)
//   cuBLAS TF32         FP32 in and out, but the multiplies use tensor cores at TF32 (10-bit mantissa)
//   cuBLAS FP16->FP32   FP16 inputs, FP32 accumulate: the standard "mixed precision" GEMM
//   cuBLAS BF16->FP32   BF16 inputs (8-bit exponent, 7-bit mantissa), FP32 accumulate
//   cuBLAS FP16->FP16   FP16 inputs, FP16 accumulate
//   WMMA naive          our own: one warp per 16 x 16 tile, fragments loaded straight from global memory
//   WMMA tiled          our own: 64 x 64 block tile staged in shared memory, 4 warps of 32 x 32 each
//
// Build & run:  nvcc -O2 -arch=native -DUSE_CUBLAS tensor_cores.cu -o tensor_cores -lcublas && ./tensor_cores
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../common.cuh"
using namespace nvcuda;

__global__ void to_half(const float* x, half* y, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
__global__ void to_bf16(const float* x, __nv_bfloat16* y, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}
__global__ void half_to_float(const half* x, float* y, size_t n) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

// One warp computes one 16 x 16 tile of C. A warp-wide fragment holds a 16 x 16
// tile spread over the 32 lanes' registers; mma_sync multiplies two of them on a tensor core.
__global__ void wmma_naive(const half* A, const half* B, float* C, int N) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tiles = N / 16;
    int tr = warp / tiles, tc = warp % tiles;
    if (tr >= tiles) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a, A + tr * 16 * N + k, N);
        wmma::load_matrix_sync(b, B + k * N + tc * 16, N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + tr * 16 * N + tc * 16, c, N, wmma::mem_row_major);
}

// 64 x 64 block tile, 4 warps in a 2 x 2 arrangement, each computing 32 x 32 (2 x 2 fragments).
// A and B tiles (64 x 32 and 32 x 64 halves) are staged in shared memory and reused by all warps.
__global__ void wmma_tiled(const half* A, const half* B, float* C, int N) {
    constexpr int BM = 64, BN = 64, BK = 32, PAD = 8;            // pad rows to avoid bank conflicts
    __shared__ half As[BM][BK + PAD], Bs[BK][BN + PAD];
    int warp = threadIdx.x / 32, wr = warp / 2, wc = warp % 2;
    int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(c[i][j], 0.0f);
    for (int k0 = 0; k0 < N; k0 += BK) {
        // 128 threads copy 64 x 32 + 32 x 64 halves, 8 halves (16 bytes) at a time
        for (int i = threadIdx.x; i < BM * BK / 8; i += 128) {
            int r = i / (BK / 8), cc = (i % (BK / 8)) * 8;
            *reinterpret_cast<int4*>(&As[r][cc]) = *reinterpret_cast<const int4*>(&A[(row0 + r) * N + k0 + cc]);
        }
        for (int i = threadIdx.x; i < BK * BN / 8; i += 128) {
            int r = i / (BN / 8), cc = (i % (BN / 8)) * 8;
            *reinterpret_cast<int4*>(&Bs[r][cc]) = *reinterpret_cast<const int4*>(&B[(k0 + r) * N + col0 + cc]);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], &As[wr * 32 + i * 16][kk], BK + PAD);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Bs[kk][wc * 32 + j * 16], BN + PAD);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(C + (row0 + wr * 32 + i * 16) * N + col0 + wc * 32 + j * 16, c[i][j], N, wmma::mem_row_major);
}

int main() {
    print_device();
    const int N = 4096;
    const size_t nn = size_t(N) * N;
    float* A = random_matrix(nn, 1);
    float* B = random_matrix(nn, 2);
    float *ref, *C;
    half *Ah, *Bh, *Ch;
    __nv_bfloat16 *Ab, *Bb;
    CUDA_CHECK(cudaMalloc(&ref, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, nn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Ah, nn * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Bh, nn * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Ch, nn * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&Ab, nn * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Bb, nn * sizeof(__nv_bfloat16)));
    int cb = int((nn + 255) / 256);
    to_half<<<cb, 256>>>(A, Ah, nn); to_half<<<cb, 256>>>(B, Bh, nn);
    to_bf16<<<cb, 256>>>(A, Ab, nn); to_bf16<<<cb, 256>>>(B, Bb, nn);

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    float t_fp32 = time_ms([&] { sgemm_ref(h, N, N, N, A, B, ref); }, 10);
    float ref_max;
    {
        std::vector<float> r(nn);
        CUDA_CHECK(cudaMemcpy(r.data(), ref, nn * sizeof(float), cudaMemcpyDeviceToHost));
        ref_max = 0; for (float v : r) ref_max = std::fmax(ref_max, std::fabs(v));
    }
    const float one = 1.0f, zero = 0.0f;
    const half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);

    std::printf("\n%d x %d x %d matmul, inputs uniform in [-1, 1]; largest |C| = %.1f\n", N, N, N, ref_max);
    std::printf("  %-20s %9s %10s %10s %14s\n", "", "time", "TFLOP/s", "vs FP32", "max |error|");
    auto report = [&](const char* name, float ms, bool check) {
        std::printf("  %-20s %6.2f ms %10.2f %9.1fx", name, ms, gflops(N, N, N, ms) / 1e3, t_fp32 / ms);
        if (check) std::printf(" %14.1e", max_abs_diff(C, ref, nn));
        std::printf("\n");
    };
    report("cuBLAS FP32", t_fp32, false);

    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    report("cuBLAS TF32", time_ms([&] { sgemm_ref(h, N, N, N, A, B, C); }, 10), true);
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    report("cuBLAS FP16->FP32", time_ms([&] {
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, Bh, CUDA_R_16F, N, Ah, CUDA_R_16F, N,
                                  &zero, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }, 10), true);
    report("cuBLAS BF16->FP32", time_ms([&] {
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, Bb, CUDA_R_16BF, N, Ab, CUDA_R_16BF, N,
                                  &zero, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }, 10), true);
    float t16 = time_ms([&] {
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one_h, Bh, CUDA_R_16F, N, Ah, CUDA_R_16F, N,
                                  &zero_h, Ch, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    }, 10);
    half_to_float<<<cb, 256>>>(Ch, C, nn);
    report("cuBLAS FP16->FP16", t16, true);

    report("WMMA naive", time_ms([&] { wmma_naive<<<(N / 16) * (N / 16) / 4, 128>>>(Ah, Bh, C, N); }, 5), true);
    report("WMMA tiled", time_ms([&] { wmma_tiled<<<dim3(N / 64, N / 64), 128>>>(Ah, Bh, C, N); }, 5), true);
    CUDA_CHECK_LAUNCH();

    cublasDestroy(h);
    cudaFree(A); cudaFree(B); cudaFree(ref); cudaFree(C); cudaFree(Ah); cudaFree(Bh); cudaFree(Ch); cudaFree(Ab); cudaFree(Bb);
}
