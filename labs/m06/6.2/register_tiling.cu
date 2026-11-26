// Lab 6.2 — Register tiling: the tile inside the tile
//
// The 6.1 tiled kernel computes one output per thread, so every multiply-add
// needs two shared-memory loads. Here each thread computes a strip, then a
// square, of outputs held in registers, and reuses every value it loads.
//
//   tiled (6.1)  32 x 32 block tile, 1 output per thread           2 smem loads per FMA
//   1D tiling    64 x 64 block tile, 8 outputs per thread (column)  9 smem loads per 8 FMAs
//   2D tiling    128 x 128 block tile, 8 x 8 outputs per thread     16 smem loads per 64 FMAs
//   2D + float4  as 2D, with 16-byte loads and A stored transposed in shared memory
//
// Build & run:  nvcc -O2 -arch=native -DUSE_CUBLAS register_tiling.cu -o register_tiling -lcublas && ./register_tiling
#include "../common.cuh"

// ---- the 6.1 tiled kernel, for comparison ------------------------------------------------------
__global__ void mm_tiled(const float* A, const float* B, float* C, int N) {
    constexpr int T = 32;
    __shared__ float As[T][T], Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += T) {
        As[ty][tx] = A[row * N + k0 + tx];
        Bs[ty][tx] = B[(k0 + ty) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---- 1D register tiling: each thread computes TM outputs down one column --------------------------
template <int BM, int BN, int BK, int TM>
__global__ void mm_1d(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int threadCol = threadIdx.x % BN, threadRow = threadIdx.x / BN;   // threadRow < BM / TM
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int rA = threadIdx.x / BK, cA = threadIdx.x % BK;   // BM*BK == threads: one load each
    const int rB = threadIdx.x / BN, cB = threadIdx.x % BN;   // BK*BN == threads
    float acc[TM] = {0.0f};
    for (int k0 = 0; k0 < N; k0 += BK) {
        As[rA * BK + cA] = A[rA * N + cA];
        Bs[rB * BN + cB] = B[rB * N + cB];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float b = Bs[k * BN + threadCol];                  // one B value, reused TM times
            #pragma unroll
            for (int r = 0; r < TM; r++) acc[r] += As[(threadRow * TM + r) * BK + k] * b;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TM; r++) C[(threadRow * TM + r) * N + threadCol] = acc[r];
}

// ---- 2D register tiling: each thread computes a TM x TN square -------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) mm_2d(const float* A, const float* B, float* C, int N) {
    constexpr int NT = (BM * BN) / (TM * TN);
    __shared__ float As[BM * BK], Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int rA = threadIdx.x / BK, cA = threadIdx.x % BK, strideA = NT / BK;
    const int rB = threadIdx.x / BN, cB = threadIdx.x % BN, strideB = NT / BN;
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];
    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int o = 0; o < BM; o += strideA) As[(rA + o) * BK + cA] = A[(rA + o) * N + cA];
        for (int o = 0; o < BK; o += strideB) Bs[(rB + o) * BN + cB] = B[(rB + o) * N + cB];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) regM[i] = As[(threadRow * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; j++) regN[j] = Bs[k * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i * TN + j] += regM[i] * regN[j];   // an outer product
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
}

// ---- 2D + float4: 16-byte global loads, A transposed in shared memory -----------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) mm_2d_vec(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BK * BM], Bs[BK * BN];   // As is stored k-major: As[k][m]
    const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int rA = threadIdx.x / (BK / 4), cA = threadIdx.x % (BK / 4);   // one float4 of A each
    const int rB = threadIdx.x / (BN / 4), cB = threadIdx.x % (BN / 4);   // one float4 of B each
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];
    for (int k0 = 0; k0 < N; k0 += BK) {
        float4 a = *reinterpret_cast<const float4*>(&A[rA * N + cA * 4]);
        As[(cA * 4 + 0) * BM + rA] = a.x;            // transpose on the way in, so the inner loop
        As[(cA * 4 + 1) * BM + rA] = a.y;            // reads TM consecutive floats of A
        As[(cA * 4 + 2) * BM + rA] = a.z;
        As[(cA * 4 + 3) * BM + rA] = a.w;
        *reinterpret_cast<float4*>(&Bs[rB * BN + cB * 4]) = *reinterpret_cast<const float4*>(&B[rB * N + cB * 4]);
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) regM[i] = As[k * BM + threadRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) regN[j] = Bs[k * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j += 4)
            *reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j]) =
                make_float4(acc[i * TN + j], acc[i * TN + j + 1], acc[i * TN + j + 2], acc[i * TN + j + 3]);
}

template <class K>
void info(K kernel, int threads, int& regs, int& blocks_per_sm) {
    cudaFuncAttributes a;
    CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
    regs = a.numRegs;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, 0));
}

int main() {
    print_device();
    const int N = 4096;
    const size_t nn = size_t(N) * N;
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
    mm_tiled<<<dim3(N / 32, N / 32), dim3(32, 32)>>>(A, B, ref, N);
    float t_ref = 0;
#endif

    std::printf("\nC = A * B, %d x %d FP32:\n", N, N);
    std::printf("  %-14s %9s %9s %10s %6s %11s %10s\n", "kernel", "time", "GFLOP/s", "of cuBLAS", "regs", "blocks/SM", "max error");
    auto report = [&](const char* name, float ms, int regs, int bps) {
        float err = max_abs_diff(C, ref, nn);
        std::printf("  %-14s %6.2f ms %9.0f %9.1f%% %6d %11d %10.1e %s\n", name, ms, gflops(N, N, N, ms),
                    t_ref > 0 ? 100.0 * t_ref / ms : 0.0, regs, bps, err, err < 1e-2 ? "PASS" : "FAIL");
    };
    int regs, bps;

    info(mm_tiled, 1024, regs, bps);
    report("tiled (6.1)", time_ms([&] { mm_tiled<<<dim3(N / 32, N / 32), dim3(32, 32)>>>(A, B, C, N); }, 5), regs, bps);

    auto k1 = mm_1d<64, 64, 8, 8>;
    info(k1, 512, regs, bps);
    report("1D tiling", time_ms([&] { k1<<<dim3(N / 64, N / 64), 512>>>(A, B, C, N); }, 5), regs, bps);

    auto k2 = mm_2d<128, 128, 8, 8, 8>;
    info(k2, 256, regs, bps);
    report("2D tiling", time_ms([&] { k2<<<dim3(N / 128, N / 128), 256>>>(A, B, C, N); }, 5), regs, bps);

    auto k3 = mm_2d_vec<128, 128, 8, 8, 8>;
    info(k3, 256, regs, bps);
    report("2D + float4", time_ms([&] { k3<<<dim3(N / 128, N / 128), 256>>>(A, B, C, N); }, 5), regs, bps);
    CUDA_CHECK_LAUNCH();
#ifdef USE_CUBLAS
    std::printf("  %-14s %6.2f ms %9.0f %9.1f%%\n", "cuBLAS", t_ref, gflops(N, N, N, t_ref), 100.0);
    cublasDestroy(h);
#endif
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ref);
}
