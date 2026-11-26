// Lab 6.4 — Shared-memory bank conflicts
//
// Shared memory is split into 32 banks, each 4 bytes wide; word i lives in bank i % 32.
// A warp's 32 loads finish in one pass only if they hit 32 different banks (or the
// same word, which is broadcast). Otherwise the hardware splits the access into as
// many passes as the busiest bank needs.
//
// Each lane reads s[lane * stride] over and over. Predicted passes = gcd(stride, 32)
// (and 1 for stride 0, a broadcast). The program measures the time per access.
//
// Build & run:  nvcc -O2 -arch=native bank_conflicts.cu -o bank_conflicts && ./bank_conflicts
#include <numeric>
#include "../common.cuh"

constexpr int WORDS = 4096;   // 16 KB of shared memory
constexpr int ITERS = 4096;

__global__ void read_strided(float* out, int stride) {
    __shared__ float s[WORDS];
    for (int i = threadIdx.x; i < WORDS; i += blockDim.x) s[i] = float(i);
    __syncthreads();
    volatile float* vs = s;                         // volatile: really do every load
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int idx = (lane * stride + warp) % WORDS;       // + warp: different warps, same pattern
    float acc = 0.0f;
    #pragma unroll 16
    for (int it = 0; it < ITERS; it++) acc += vs[idx];
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main() {
    print_device();
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = sms * 4, threads = 256;
    float* out;
    CUDA_CHECK(cudaMalloc(&out, blocks * threads * sizeof(float)));

    struct Case { int stride; const char* what; };
    Case cases[] = {
        {0, "all lanes read one word (broadcast)"},
        {1, "consecutive words"},
        {2, "every other word"},
        {3, "odd stride"},
        {4, ""},
        {8, ""},
        {16, ""},
        {32, "a column of float[32][32]"},
        {33, "a column of float[32][33] (padded)"},
    };
    float ms[16];
    int n = sizeof(cases) / sizeof(cases[0]);
    for (int i = 0; i < n; i++) {
        ms[i] = time_ms([&] { read_strided<<<blocks, threads>>>(out, cases[i].stride); }, 10);
        CUDA_CHECK_LAUNCH();
    }
    std::printf("\nEach lane reads s[lane * stride], %d times; %d blocks of %d threads:\n", ITERS, blocks, threads);
    std::printf("  %6s %10s %10s %10s\n", "stride", "predicted", "time", "vs stride 1");
    for (int i = 0; i < n; i++) {
        int ways = cases[i].stride == 0 ? 1 : std::gcd(cases[i].stride, 32);
        std::printf("  %6d %8d-way %7.3f ms %9.1fx   %s\n", cases[i].stride, ways, ms[i], ms[i] / ms[1], cases[i].what);
    }
    cudaFree(out);
}
