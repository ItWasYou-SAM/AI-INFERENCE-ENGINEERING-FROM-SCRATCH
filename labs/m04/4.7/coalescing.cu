// Lab 4.7 — Coalesced vs strided access
//
// Each thread copies one float, reading from index (i * stride) % n.
// stride 1: the 32 lanes of a warp read 32 adjacent floats (128 bytes).
// stride 32: every lane hits a different 32-byte sector.
// Then a matrix transpose: naive (strided writes) vs shared-memory tiled.
//
// Build & run:  nvcc -O2 -arch=native coalescing.cu -o coalescing && ./coalescing
#include <vector>
#include "../common.cuh"

__global__ void strided_copy(const float* in, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[(size_t(i) * stride) % n];
}

constexpr int TILE = 32;

__global__ void transpose_naive(const float* in, float* out, int N) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += 8)                    // 32 x 8 threads, 4 rows each
        out[x * N + (y + j)] = in[(y + j) * N + x];      // read coalesced, write strided by N
}

__global__ void transpose_tiled(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];               // +1 avoids bank conflicts (chapter 6.4)
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += 8) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;                 // swap block coordinates
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += 8) out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
    print_device();
    const int n = 1 << 25;                               // 32 M floats = 128 MB
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 0, n * sizeof(float)));

    std::printf("\nStrided read, coalesced write (%d MB each way):\n", int(n * 4 >> 20));
    std::printf("  %6s  %8s  %s\n", "stride", "GB/s", "useful bytes per 32-byte sector");
    for (int stride : {1, 2, 4, 8, 16, 32, 64}) {
        strided_copy<<<n / 256, 256>>>(in, out, n, stride);   // warm-up
        GpuTimer t;
        t.start();
        for (int r = 0; r < 5; r++) strided_copy<<<n / 256, 256>>>(in, out, n, stride);
        float ms = t.stop_ms() / 5;
        CUDA_CHECK_LAUNCH();
        int useful = stride >= 8 ? 4 : 32 / stride;
        std::printf("  %6d  %8.1f  %2d of 32\n", stride, 2.0 * n * 4 / ms / 1e6, useful);
    }

    const int N = 4096;
    std::vector<float> h(size_t(N) * N), back(size_t(N) * N);
    for (size_t i = 0; i < h.size(); i++) h[i] = float(i % 1000);
    float *a, *b;
    CUDA_CHECK(cudaMalloc(&a, h.size() * 4));
    CUDA_CHECK(cudaMalloc(&b, h.size() * 4));
    CUDA_CHECK(cudaMemcpy(a, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    dim3 grid(N / TILE, N / TILE), block(TILE, 8);
    std::printf("\nTranspose %d x %d floats:\n", N, N);
    auto run = [&](auto kernel, const char* name) {
        kernel<<<grid, block>>>(a, b, N);
        GpuTimer t;
        t.start();
        for (int r = 0; r < 5; r++) kernel<<<grid, block>>>(a, b, N);
        float ms = t.stop_ms() / 5;
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(back.data(), b, h.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < N; r += 97)
            for (int c = 0; c < N; c += 89) bad += back[size_t(c) * N + r] != h[size_t(r) * N + c];
        std::printf("  %-6s %7.3f ms  %6.1f GB/s  %s\n", name, ms, 2.0 * h.size() * 4 / ms / 1e6, bad ? "FAIL" : "PASS");
    };
    run(transpose_naive, "naive");
    run(transpose_tiled, "tiled");

    cudaFree(in); cudaFree(out); cudaFree(a); cudaFree(b);
}
