// Lab 4.5 — Race conditions and correctness
//
// Count how many elements are positive, three ways:
//   naive   : every thread does (*count)++     -> a data race, wrong answer
//   atomic  : every thread does atomicAdd       -> correct, but contended
//   block   : count per block in shared memory, one atomicAdd per block
//
// Build & run:  nvcc -O2 -arch=native race.cu -o race && ./race
// Note: compute-sanitizer --tool racecheck finds races in *shared* memory (try it on
// lab 4.4's no-sync kernel). Global-memory races like count_naive need careful reasoning.
#include <vector>
#include "../common.cuh"

__global__ void count_naive(const float* x, int n, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > 0) (*count)++;         // read, add, write: not one step
}

__global__ void count_atomic(const float* x, int n, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > 0) atomicAdd(count, 1); // hardware makes read-add-write indivisible
}

__global__ void count_block(const float* x, int n, int* count) {
    __shared__ int local;
    if (threadIdx.x == 0) local = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] > 0) atomicAdd(&local, 1); // shared-memory atomics: cheap, block-local
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(count, local); // one global atomic per block
}

int main() {
    print_device();
    const int n = 1 << 24;
    std::vector<float> h(n);
    int expect = 0;
    for (int i = 0; i < n; i++) { h[i] = (i % 3 == 0) ? 1.0f : -1.0f; expect += h[i] > 0; }

    float* x;
    int* count;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&](auto kernel, const char* name) {
        kernel<<<n / 256, 256>>>(x, n, count);   // warm-up
        CUDA_CHECK(cudaMemset(count, 0, sizeof(int)));
        GpuTimer t;
        t.start();
        kernel<<<n / 256, 256>>>(x, n, count);
        float ms = t.stop_ms();
        CUDA_CHECK_LAUNCH();
        int got;
        CUDA_CHECK(cudaMemcpy(&got, count, sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("  %-8s count = %9d  (%s)  %7.3f ms\n", name, got, got == expect ? "correct" : "WRONG", ms);
    };
    std::printf("\nCounting positives among %d elements (expected %d):\n", n, expect);
    run(count_naive, "naive");
    run(count_atomic, "atomic");
    run(count_block, "block");

    cudaFree(x); cudaFree(count);
}
