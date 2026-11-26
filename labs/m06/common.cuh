// Shared helpers for the Module 6 labs: the Module 5 helpers (and through them,
// Module 4's), plus matrix setup, a cuBLAS reference and an error check.
//
// All matrices are row-major. Labs that use cuBLAS link with -lcublas.
#pragma once
#include <cmath>
#include <random>
#include <vector>
#include "../m05/common.cuh"

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#define CUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        cublasStatus_t s_ = (call);                                                     \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                              \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", int(s_), __FILE__, __LINE__); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

// C = A * B for row-major M x K and K x N matrices. cuBLAS is column-major, and a
// row-major matrix is the transpose of a column-major one, so ask cuBLAS for
// C^T = B^T * A^T: pass B first, then A, with the dimensions swapped.
inline void sgemm_ref(cublasHandle_t h, int M, int N, int K, const float* A, const float* B, float* C) {
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
}
#endif

// A device buffer of n floats filled with uniform values in [-1, 1).
inline float* random_matrix(size_t n, unsigned seed) {
    std::vector<float> h(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    for (auto& v : h) v = U(rng);
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// Largest absolute difference between two device buffers.
inline float max_abs_diff(const float* a, const float* b, size_t n) {
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b, n * sizeof(float), cudaMemcpyDeviceToHost));
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) m = std::fmax(m, std::fabs(ha[i] - hb[i]));
    return m;
}

inline double gflops(int M, int N, int K, float ms) { return 2.0 * M * N * K / (ms * 1e6); }
