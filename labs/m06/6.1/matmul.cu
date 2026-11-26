// Lab 6.1 — Naive → coalesced → shared-memory tiled matmul
//
// C = A * B for square N x N FP32 matrices, three ways, against cuBLAS:
//
//   naive      one thread per output; threadIdx.x walks down a *column* of C,
//              so neighbouring threads read rows of A that are N floats apart
//   coalesced  the same, but threadIdx.x walks along a *row* of C: a warp reads
//              one element of A (broadcast) and 32 neighbouring elements of B
//   tiled      each 32 x 32 block loads 32 x 32 tiles of A and B into shared
//              memory, then every thread reuses each loaded value 32 times
//
// Build & run:  nvcc -O2 -arch=native -DUSE_CUBLAS matmul.cu -o matmul -lcublas && ./matmul
// (No cuBLAS? Drop -DUSE_CUBLAS and -lcublas: the kernels are then checked against each other.)
#include "../common.cuh"

constexpr int TILE = 32;

__global__ void mm_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // x → row: the uncoalesced choice
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < N; k++) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

__global__ void mm_coalesced(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → column: a warp covers 32 adjacent columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < N; k++) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

__global__ void mm_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += TILE) {
        As[ty][tx] = A[row * N + k0 + tx];            // coalesced: a warp loads one row of the tile
        Bs[ty][tx] = B[(k0 + ty) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();                              // don't overwrite tiles still being read
    }
    C[row * N + col] = acc;
}

int main() {
    print_device();
    for (int N : {1024, 4096}) {
        size_t nn = size_t(N) * N;
        float* A = random_matrix(nn, 1);
        float* B = random_matrix(nn, 2);
        float *C, *ref;
        CUDA_CHECK(cudaMalloc(&C, nn * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ref, nn * sizeof(float)));

#ifdef USE_CUBLAS
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        float t_ref = time_ms([&] { sgemm_ref(h, N, N, N, A, B, ref); }, 10);
#else
        dim3 g(N / TILE, N / TILE), b(TILE, TILE);
        mm_tiled<<<g, b>>>(A, B, ref, N);
        float t_ref = 0;
#endif
        std::printf("\nC = A * B, %d x %d FP32 (%.1f GFLOP):\n", N, N, 2.0 * N * N * N / 1e9);
        std::printf("  %-12s %10s %12s %10s %12s\n", "kernel", "time", "GFLOP/s", "of cuBLAS", "max |error|");
        auto report = [&](const char* name, float ms) {
            float err = max_abs_diff(C, ref, nn);
            std::printf("  %-12s %7.2f ms %12.0f %9.1f%% %12.1e %s\n", name, ms, gflops(N, N, N, ms),
                        t_ref > 0 ? 100.0 * t_ref / ms : 0.0, err, err < 1e-2 ? "PASS" : "FAIL");
        };
        dim3 b(TILE, TILE), g(N / TILE, N / TILE);
        int reps = N > 2048 ? 3 : 10;
        report("naive", time_ms([&] { mm_naive<<<g, b>>>(A, B, C, N); }, reps));
        report("coalesced", time_ms([&] { mm_coalesced<<<g, b>>>(A, B, C, N); }, reps));
        report("tiled", time_ms([&] { mm_tiled<<<g, b>>>(A, B, C, N); }, reps));
        CUDA_CHECK_LAUNCH();
#ifdef USE_CUBLAS
        std::printf("  %-12s %7.2f ms %12.0f %9.1f%%\n", "cuBLAS", t_ref, gflops(N, N, N, t_ref), 100.0);
        cublasDestroy(h);
#endif
        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ref);
    }
}
