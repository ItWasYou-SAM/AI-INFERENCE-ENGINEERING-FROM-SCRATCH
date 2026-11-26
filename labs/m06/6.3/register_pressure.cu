// Lab 6.3 — Register pressure, spilling and the occupancy trade-off
//
// The 2D register-tiled matmul from 6.2, built several ways:
//   - with different per-thread tiles (TM x TN): bigger tiles reuse more, but need more registers
//   - with __launch_bounds__(threads, MINB): asks the compiler to fit MINB blocks per SM,
//     which caps registers per thread, and forces spills to local memory when the cap is too low
//
// For each build the program asks the runtime how many registers each thread uses,
// how many bytes it spilled to local memory, and how many blocks fit per SM.
// Compile with -Xptxas -v to see the same numbers from the compiler.
//
// Build & run:  nvcc -O2 -arch=native -DUSE_CUBLAS register_pressure.cu -o register_pressure -lcublas && ./register_pressure
#include "../common.cuh"

template <int BM, int BN, int BK, int TM, int TN, int MINB>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), MINB)
mm(const float* A, const float* B, float* C, int N) {
    constexpr int NT = (BM * BN) / (TM * TN);
    __shared__ float As[BK * BM], Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * N;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];
    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK / 4; i += NT) {        // A: BM rows of BK/4 float4s
            int r = i / (BK / 4), c = i % (BK / 4);
            float4 a = *reinterpret_cast<const float4*>(&A[r * N + c * 4]);
            As[(c * 4 + 0) * BM + r] = a.x; As[(c * 4 + 1) * BM + r] = a.y;
            As[(c * 4 + 2) * BM + r] = a.z; As[(c * 4 + 3) * BM + r] = a.w;
        }
        for (int i = threadIdx.x; i < BK * BN / 4; i += NT) {        // B: BK rows of BN/4 float4s
            int r = i / (BN / 4), c = i % (BN / 4);
            *reinterpret_cast<float4*>(&Bs[r * BN + c * 4]) = *reinterpret_cast<const float4*>(&B[r * N + c * 4]);
        }
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

int main() {
    print_device();
    int max_threads_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0));
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
    sgemm_ref(h, N, N, N, A, B, ref);
    float t_ref = time_ms([&] { sgemm_ref(h, N, N, N, A, B, ref); }, 10);
#else
    mm<128, 128, 8, 8, 8, 1><<<dim3(N / 128, N / 128), 256>>>(A, B, ref, N);
    float t_ref = 0;
#endif

    std::printf("\n%d x %d FP32 matmul, one kernel built seven ways:\n", N, N);
    std::printf("  %-26s %5s %8s %10s %10s %9s %9s\n", "per-thread tile, bound", "regs", "spilled", "blocks/SM", "occupancy", "time", "GFLOP/s");
    auto run = [&](const char* name, auto kernel, int BM, int BN, int threads) {
        cudaFuncAttributes a;
        CUDA_CHECK(cudaFuncGetAttributes(&a, kernel));
        int bps;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, 0));
        dim3 g(N / BN, N / BM);
        float ms = time_ms([&] { kernel<<<g, threads>>>(A, B, C, N); }, 5);
        CUDA_CHECK_LAUNCH();
        float err = max_abs_diff(C, ref, nn);
        char spill[16];
        std::snprintf(spill, sizeof spill, "%zu B", a.localSizeBytes);
        std::printf("  %-26s %5d %8s %10d %9.0f%% %6.2f ms %9.0f %s\n", name, a.numRegs, spill, bps,
                    100.0 * bps * threads / max_threads_sm, ms, gflops(N, N, N, ms), err < 1e-2 ? "" : "FAIL");
    };
    run("4x4, no bound", mm<64, 64, 8, 4, 4, 1>, 64, 64, 256);
    run("8x4, no bound", mm<128, 64, 8, 8, 4, 1>, 128, 64, 256);
    run("8x8, no bound", mm<128, 128, 8, 8, 8, 1>, 128, 128, 256);
    run("8x8, min 2 blocks/SM", mm<128, 128, 8, 8, 8, 2>, 128, 128, 256);
    run("8x8, min 3 blocks/SM", mm<128, 128, 8, 8, 8, 3>, 128, 128, 256);
    run("8x8, min 4 blocks/SM", mm<128, 128, 8, 8, 8, 4>, 128, 128, 256);
    run("16x8, no bound", mm<128, 128, 8, 16, 8, 1>, 128, 128, 128);
#ifdef USE_CUBLAS
    std::printf("  %-26s %5s %8s %10s %10s %6.2f ms %9.0f\n", "cuBLAS", "", "", "", "", t_ref, gflops(N, N, N, t_ref));
    cublasDestroy(h);
#endif
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ref);
}
