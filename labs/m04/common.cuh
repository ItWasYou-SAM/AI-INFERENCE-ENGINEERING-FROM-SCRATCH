// Shared helpers for the Module 4 labs.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Every CUDA API call returns an error code. Check all of them: a failed call
// otherwise shows up much later as a confusing wrong answer.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                         __FILE__, __LINE__, cudaGetErrorString(err_));         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Kernel launches don't return an error code; check right after launching.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Times GPU work with CUDA events, which are recorded on the GPU's own timeline.
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t s = 0) { cudaEventRecord(start_, s); }
    float stop_ms(cudaStream_t s = 0) {
        float ms = 0;
        cudaEventRecord(stop_, s);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

inline void print_device() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("GPU: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
}
