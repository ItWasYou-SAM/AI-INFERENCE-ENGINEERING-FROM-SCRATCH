// Lab 5.5 — Histograms and privatization
//
// A 256-bin histogram of 64 M bytes, four ways, on three inputs:
//   uniform     every bin equally likely
//   skewed      90% of the bytes are 0 (think padding tokens, or one hot expert)
//   constant    every byte is the same: the worst case for contention
//
//   global       one global atomicAdd per byte
//   private      a 256-bin histogram per block in shared memory, merged at the end
//   private+     the same, plus a grid-stride loop and 4 bytes per load, so far fewer blocks merge
//   per-warp     each warp gets its own shared-memory copy, so the 8 warps of a block
//                stop contending with each other
//
// Build & run:  nvcc -O2 -arch=native histogram.cu -o histogram && ./histogram
#include <random>
#include <vector>
#include "../common.cuh"

constexpr int BINS = 256, BLOCK = 256, WARPS = BLOCK / 32;

__global__ void hist_global(const unsigned char* x, unsigned* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[x[i]], 1u);
}

__global__ void hist_private(const unsigned char* x, unsigned* hist, int n) {
    __shared__ unsigned local[BINS];
    local[threadIdx.x] = 0;                     // BLOCK == BINS: one bin per thread to clear
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&local[x[i]], 1u);     // shared-memory atomic: on-chip, cheap
    __syncthreads();
    if (local[threadIdx.x]) atomicAdd(&hist[threadIdx.x], local[threadIdx.x]);   // merge
}

__global__ void hist_private_plus(const unsigned char* x, unsigned* hist, int n) {
    __shared__ unsigned local[BINS];
    local[threadIdx.x] = 0;
    __syncthreads();
    const uchar4* x4 = reinterpret_cast<const uchar4*>(x);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += gridDim.x * blockDim.x) {
        uchar4 q = x4[i];
        atomicAdd(&local[q.x], 1u); atomicAdd(&local[q.y], 1u);
        atomicAdd(&local[q.z], 1u); atomicAdd(&local[q.w], 1u);
    }
    __syncthreads();
    if (local[threadIdx.x]) atomicAdd(&hist[threadIdx.x], local[threadIdx.x]);
}

__global__ void hist_per_warp(const unsigned char* x, unsigned* hist, int n) {
    __shared__ unsigned local[WARPS][BINS];     // 8 KB: one private copy per warp
    for (int w = 0; w < WARPS; w++) local[w][threadIdx.x] = 0;
    __syncthreads();
    unsigned* mine = local[threadIdx.x / 32];
    const uchar4* x4 = reinterpret_cast<const uchar4*>(x);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += gridDim.x * blockDim.x) {
        uchar4 q = x4[i];
        atomicAdd(&mine[q.x], 1u); atomicAdd(&mine[q.y], 1u);
        atomicAdd(&mine[q.z], 1u); atomicAdd(&mine[q.w], 1u);
    }
    __syncthreads();
    unsigned total = 0;
    for (int w = 0; w < WARPS; w++) total += local[w][threadIdx.x];
    if (total) atomicAdd(&hist[threadIdx.x], total);
}

int main() {
    print_device();
    const int n = 1 << 26;                       // 64 M bytes (a multiple of 4)
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int few_blocks = sms * 6;

    std::mt19937 rng(1);
    std::vector<unsigned char> h(n);
    unsigned char* x;
    unsigned* hist;
    CUDA_CHECK(cudaMalloc(&x, n));
    CUDA_CHECK(cudaMalloc(&hist, BINS * sizeof(unsigned)));

    struct K { const char* name; void (*k)(const unsigned char*, unsigned*, int); bool few; };
    K ks[] = {{"global", hist_global, false},
              {"private", hist_private, false},
              {"private+", hist_private_plus, true},
              {"per-warp", hist_per_warp, true}};

    std::printf("\n256-bin histogram of %d bytes. Time in ms (GB/s):\n", n);
    std::printf("  %-10s", "input");
    for (auto& k : ks) std::printf(" %20s", k.name);
    std::printf("\n");

    for (int dist = 0; dist < 3; dist++) {
        const char* names[] = {"uniform", "skewed", "constant"};
        for (auto& v : h) {
            if (dist == 0) v = rng() & 255;
            else if (dist == 1) v = (rng() % 10 == 0) ? (rng() & 255) : 0;
            else v = 42;
        }
        std::vector<unsigned> expect(BINS, 0), got(BINS);
        for (auto v : h) expect[v]++;
        CUDA_CHECK(cudaMemcpy(x, h.data(), n, cudaMemcpyHostToDevice));

        std::printf("  %-10s", names[dist]);
        for (auto& k : ks) {
            int blocks = k.few ? few_blocks : n / BLOCK;
            float ms = time_ms([&] {
                cudaMemsetAsync(hist, 0, BINS * sizeof(unsigned));
                k.k<<<blocks, BLOCK>>>(x, hist, n);
            }, 10);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(got.data(), hist, BINS * sizeof(unsigned), cudaMemcpyDeviceToHost));
            char cell[32];
            std::snprintf(cell, sizeof cell, "%.3f (%5.1f)%s", ms, n / ms / 1e6, got == expect ? "" : " FAIL");
            std::printf(" %20s", cell);
        }
        std::printf("\n");
    }
    cudaFree(x); cudaFree(hist);
}
