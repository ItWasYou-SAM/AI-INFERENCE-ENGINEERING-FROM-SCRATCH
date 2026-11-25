// Lab 4.4 — Shared memory and __syncthreads
//
// Each block sums 256 elements using shared memory and a tree reduction.
// Run once correctly and once with the __syncthreads() removed, to see
// what a missing barrier does.
//
// Build & run:  nvcc -O2 -arch=native shared_sum.cu -o shared_sum && ./shared_sum
// Detect the race:  compute-sanitizer --tool racecheck ./shared_sum
#include <vector>
#include "../common.cuh"

constexpr int BLOCK = 256;

template <bool SYNC>
__global__ void block_sum(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];                  // one copy per block, visible to all its threads
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;            // 1. each thread loads one element
    if (SYNC) __syncthreads();                  //    wait until every load has landed

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {   // 2. tree: 128, 64, ..., 1
        if (tid < stride) s[tid] += s[tid + stride];
        if (SYNC) __syncthreads();              //    wait for this level before the next
    }
    if (tid == 0) out[blockIdx.x] = s[0];       // 3. thread 0 writes the block's total
}

int main() {
    print_device();
    const int n = 1 << 24;
    const int blocks = n / BLOCK;
    std::vector<float> h(n);
    for (int i = 0; i < n; i++) h[i] = float(i % 7);   // small integers: sums are exact in FP32

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> expect(blocks, 0.0f), got(blocks);
    for (int b = 0; b < blocks; b++)
        for (int t = 0; t < BLOCK; t++) expect[b] += h[b * BLOCK + t];

    auto run = [&](auto kernel, const char* name) {
        int wrong = 0;
        for (int trial = 0; trial < 20; trial++) {
            kernel<<<blocks, BLOCK>>>(in, out, n);
            CUDA_CHECK_LAUNCH();
            CUDA_CHECK(cudaMemcpy(got.data(), out, blocks * sizeof(float), cudaMemcpyDeviceToHost));
            for (int b = 0; b < blocks; b++) wrong += got[b] != expect[b];
        }
        std::printf("  %-26s wrong block sums: %8d of %d (20 runs)\n", name, wrong, blocks * 20);
    };
    std::printf("\nSumming %d blocks of %d elements:\n", blocks, BLOCK);
    run(block_sum<true>, "with __syncthreads()");
    run(block_sum<false>, "without __syncthreads()");

    cudaFree(in); cudaFree(out);
}
