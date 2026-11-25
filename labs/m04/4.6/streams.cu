// Lab 4.6 — Kernel launch anatomy, async execution and streams
//
// 1. Launch overhead: how long does an empty kernel launch cost?
// 2. CUDA Graphs: capture many launches once, replay them with one call.
// 3. Streams: split work into chunks so copies and kernels overlap.
//
// Build & run:  nvcc -O2 -arch=native streams.cu -o streams && ./streams
#include <chrono>
#include "../common.cuh"

__global__ void empty() {}

__global__ void busy(float* x, int n) {             // some arithmetic per element
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        for (int k = 0; k < 200; k++) v = v * 0.999f + 0.001f;
        x[i] = v;
    }
}

int main() {
    print_device();

    // 1) launch overhead
    const int launches = 20000;
    empty<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < launches; i++) empty<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n1) %d empty launches: %.2f us per launch\n", launches, us / launches);

    // 2) the same launches captured into a CUDA Graph
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    const int per_graph = 100;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < per_graph; i++) empty<<<1, 1, 0, s>>>();
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < launches / per_graph; i++) CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    std::printf("2) same kernels via a CUDA Graph: %.2f us per kernel\n", us / launches);

    // 3) copy -> compute -> copy back, serial vs chunked over 4 streams
    const int n = 1 << 25;                           // 128 MB
    const size_t bytes = size_t(n) * sizeof(float);
    float *h, *d;
    CUDA_CHECK(cudaMallocHost(&h, bytes));           // pinned: required for async copies to overlap
    CUDA_CHECK(cudaMalloc(&d, bytes));
    for (int i = 0; i < n; i++) h[i] = 1.0f;

    GpuTimer t;
    t.start();
    CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    busy<<<n / 256, 256>>>(d, n);
    CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    float serial = t.stop_ms();

    const int NS = 4, chunk = n / NS;
    cudaStream_t st[NS];
    for (auto& x : st) CUDA_CHECK(cudaStreamCreate(&x));
    t.start();
    for (int k = 0; k < NS; k++) {
        float* hp = h + size_t(k) * chunk;
        float* dp = d + size_t(k) * chunk;
        CUDA_CHECK(cudaMemcpyAsync(dp, hp, chunk * sizeof(float), cudaMemcpyHostToDevice, st[k]));
        busy<<<chunk / 256, 256, 0, st[k]>>>(dp, chunk);
        CUDA_CHECK(cudaMemcpyAsync(hp, dp, chunk * sizeof(float), cudaMemcpyDeviceToHost, st[k]));
    }
    float overlapped = t.stop_ms();   // the stop event on stream 0 waits for all streams (legacy default stream)
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("3) copy in + kernel + copy out, %zu MB:\n", bytes >> 20);
    std::printf("   one stream     : %7.2f ms\n", serial);
    std::printf("   %d streams      : %7.2f ms  (%.2fx)\n", NS, overlapped, serial / overlapped);

    for (auto& x : st) cudaStreamDestroy(x);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFreeHost(h);
    cudaFree(d);
}
