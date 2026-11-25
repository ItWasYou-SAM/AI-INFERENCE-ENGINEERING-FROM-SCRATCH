// Lab 4.1 — A kernel is not a function
//
// 1. Launch a kernel and print from a few threads (order is not guaranteed).
// 2. Show that a launch returns immediately: the CPU moves on while the GPU works.
// 3. Vector add on the GPU, checked against the CPU.
//
// Build & run:  nvcc -O2 -arch=native vector_add.cu -o vector_add && ./vector_add
#include <chrono>
#include <vector>
#include "../common.cuh"

__global__ void hello() {
    if (threadIdx.x < 2)   // keep the output short: two threads in each block
        printf("  hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// One thread per element. Each thread computes exactly one c[i].
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // guard: the last block may run past n
}

int main() {
    print_device();

    std::printf("\n1) hello<<<3, 32>>>: blocks run in any order\n");
    hello<<<3, 32>>>();
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());   // wait, so printf output is flushed

    const int n = 1 << 26;                 // 64 M floats = 256 MB per array
    const size_t bytes = size_t(n) * sizeof(float);
    std::vector<float> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; i++) { ha[i] = float(i % 1000); hb[i] = 2.0f; }

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemcpy(a, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, hb.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;   // round up
    vector_add<<<blocks, threads>>>(a, b, c, n);      // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("\n2) launches are asynchronous\n");
    auto t0 = std::chrono::steady_clock::now();
    vector_add<<<blocks, threads>>>(a, b, c, n);
    auto t1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();
    auto us = [](auto d) { return std::chrono::duration<double, std::micro>(d).count(); };
    std::printf("  launch call returned after %8.1f us\n", us(t1 - t0));
    std::printf("  kernel finished after      %8.1f us\n", us(t2 - t0));

    std::printf("\n3) vector add, %d elements, %d blocks x %d threads\n", n, blocks, threads);
    GpuTimer timer;
    timer.start();
    vector_add<<<blocks, threads>>>(a, b, c, n);
    float ms = timer.stop_ms();
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(hc.data(), c, bytes, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; i++) bad += hc[i] != ha[i] + hb[i];
    std::printf("  kernel time %.3f ms -> %.0f GB/s (3 x 256 MB moved)\n", ms, 3.0 * bytes / ms / 1e6);
    std::printf("  check against CPU: %s\n", bad ? "FAIL" : "PASS");

    cudaFree(a); cudaFree(b); cudaFree(c);
    return bad != 0;
}
