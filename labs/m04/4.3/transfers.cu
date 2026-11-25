// Lab 4.3 — Host/device memory and transfers
//
// Measure host->device and device->host bandwidth for pageable vs pinned
// host memory, and compare with on-GPU copy bandwidth.
//
// Build & run:  nvcc -O2 -arch=native transfers.cu -o transfers && ./transfers
#include <cstring>
#include "../common.cuh"

float time_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps = 10) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));   // warm-up
    GpuTimer t;
    t.start();
    for (int r = 0; r < reps; r++) CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    return t.stop_ms() / reps;
}

int main() {
    print_device();
    const size_t bytes = 256u << 20;   // 256 MB

    void* pageable = std::malloc(bytes);          // ordinary memory: can be paged out by the OS
    void* pinned;
    CUDA_CHECK(cudaMallocHost(&pinned, bytes));   // page-locked: the GPU can DMA it directly
    std::memset(pageable, 1, bytes);
    std::memset(pinned, 1, bytes);
    void *dev, *dev2;
    CUDA_CHECK(cudaMalloc(&dev, bytes));
    CUDA_CHECK(cudaMalloc(&dev2, bytes));

    auto gbps = [&](float ms) { return bytes / ms / 1e6; };
    std::printf("\nCopying %zu MB:\n", bytes >> 20);
    std::printf("  host->device, pageable : %6.1f GB/s\n", gbps(time_copy(dev, pageable, bytes, cudaMemcpyHostToDevice)));
    std::printf("  host->device, pinned   : %6.1f GB/s\n", gbps(time_copy(dev, pinned, bytes, cudaMemcpyHostToDevice)));
    std::printf("  device->host, pageable : %6.1f GB/s\n", gbps(time_copy(pageable, dev, bytes, cudaMemcpyDeviceToHost)));
    std::printf("  device->host, pinned   : %6.1f GB/s\n", gbps(time_copy(pinned, dev, bytes, cudaMemcpyDeviceToHost)));
    float d2d = time_copy(dev2, dev, bytes, cudaMemcpyDeviceToDevice);
    std::printf("  device->device         : %6.1f GB/s (read + write = %.1f GB/s of DRAM traffic)\n", gbps(d2d), 2 * gbps(d2d));

    std::printf("\nSmall transfers are dominated by fixed latency:\n");
    for (size_t sz : {size_t(4), size_t(4) << 10, size_t(1) << 20, size_t(16) << 20}) {
        float ms = time_copy(dev, pinned, sz, cudaMemcpyHostToDevice, 100);
        std::printf("  %9zu bytes: %8.1f us  -> %7.2f GB/s\n", sz, ms * 1e3, sz / ms / 1e6);
    }

    std::free(pageable);
    cudaFreeHost(pinned);
    cudaFree(dev); cudaFree(dev2);
}
