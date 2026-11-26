// Lab 6.5 — Wave quantization
//
// A GPU runs a grid in "waves": as many blocks as fit on all SMs at once, then the
// next batch. If the last wave is only partly full, the GPU is partly idle for a
// whole block-duration.
//
// Part 1: a synthetic kernel where every block spins for the same time, with one
//         block per SM. Time vs grid size is a staircase with a step every #SMs blocks.
// Part 2: the 128 x 128 register-tiled matmul from 6.2 (2 blocks per SM fit),
//         M = 1280 and K = 4096 fixed, N growing one tile at a time.
//
// Build & run:  nvcc -O2 -arch=native waves.cu -o waves && ./waves
#include "../common.cuh"

__global__ void spin(long long cycles, int* sink) {
    extern __shared__ char smem[];               // sized so only one block fits per SM
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (threadIdx.x == 0) sink[blockIdx.x] = smem[0];
}

// the 6.2 "2D + float4" kernel, unchanged (BM = BN = 128, BK = 8, 8 x 8 per thread)
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) mm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BK * BM], Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN), threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int rA = threadIdx.x / (BK / 4), cA = threadIdx.x % (BK / 4);
    const int rB = threadIdx.x / (BN / 4), cB = threadIdx.x % (BN / 4);
    float acc[TM * TN] = {0.0f}, regM[TM], regN[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        float4 a = *reinterpret_cast<const float4*>(&A[rA * K + cA * 4]);
        As[(cA * 4 + 0) * BM + rA] = a.x; As[(cA * 4 + 1) * BM + rA] = a.y;
        As[(cA * 4 + 2) * BM + rA] = a.z; As[(cA * 4 + 3) * BM + rA] = a.w;
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

int main() {
    print_device();
    int sms, smem_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));

    // Part 1
    int smem = smem_sm / 2 + 1024;               // more than half an SM's shared memory: 1 block per SM
    CUDA_CHECK(cudaFuncSetAttribute(spin, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int bps;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, spin, 128, smem));
    int* sink;
    CUDA_CHECK(cudaMalloc(&sink, 4096 * sizeof(int)));
    const long long cycles = 1000000;            // ~0.6 ms per block at ~1.6 GHz
    std::printf("\nPart 1: every block spins for %lld cycles; %d block(s) fit per SM, %d SMs:\n", cycles, bps, sms);
    std::printf("  %7s %7s %9s %12s\n", "blocks", "waves", "time", "SM busy");
    for (int b : {1, 10, 19, 20, 21, 30, 39, 40, 41, 60, 61}) {
        float ms = time_ms([&] { spin<<<b, 128, smem>>>(cycles, sink); }, 5);
        CUDA_CHECK_LAUNCH();
        int waves = (b + sms * bps - 1) / (sms * bps);
        std::printf("  %7d %7d %6.2f ms %11.0f%%\n", b, waves, ms, 100.0 * b / (waves * sms * bps));
    }

    // Part 2
    const int M = 1280, K = 4096, Nmax = 128 * 12;
    float* A = random_matrix(size_t(M) * K, 1);
    float* B = random_matrix(size_t(K) * Nmax, 2);
    float* C;
    CUDA_CHECK(cudaMalloc(&C, size_t(M) * Nmax * sizeof(float)));
    auto k = mm<128, 128, 8, 8, 8>;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k, 256, 0));
    std::printf("\nPart 2: %d x N x %d matmul, 128 x 128 tiles; %d blocks fit per SM, so %d tiles per wave:\n", M, K, bps, bps * sms);
    std::printf("  %6s %6s %6s %9s %10s %12s\n", "N", "tiles", "waves", "time", "GFLOP/s", "ms per tile");
    for (int nt = 1; nt <= 12; nt++) {
        int N = 128 * nt;
        dim3 g(N / 128, M / 128);
        // timing only: B is read as a K x N matrix from the start of its (K x Nmax) buffer
        float ms = time_ms([&] { k<<<g, 256>>>(A, B, C, M, N, K); }, 5);
        CUDA_CHECK_LAUNCH();
        int tiles = g.x * g.y, waves = (tiles + bps * sms - 1) / (bps * sms);
        std::printf("  %6d %6d %6d %6.2f ms %10.0f %12.3f\n", N, tiles, waves, ms, gflops(M, N, K, ms), ms / tiles);
    }
    cudaFree(sink); cudaFree(A); cudaFree(B); cudaFree(C);
}
