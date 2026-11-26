// Shared helpers for the Module 5 labs: the Module 4 helpers plus a few more.
#pragma once
#include "../m04/common.cuh"

// Peak DRAM bandwidth in GB/s, from the device's memory clock and bus width.
inline double peak_bandwidth_gbs() {
    int clock_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    return 2.0 * clock_khz * 1e3 * (bus_bits / 8) / 1e9;   // DDR: two transfers per clock
}

// Runs fn() a few times to warm up, then returns the median time in ms of `reps` runs.
template <class F>
float time_ms(F fn, int reps = 20) {
    for (int i = 0; i < 3; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    float t[64];
    GpuTimer timer;
    for (int r = 0; r < reps; r++) { timer.start(); fn(); t[r] = timer.stop_ms(); }
    for (int i = 1; i < reps; i++)                     // insertion sort, then take the middle
        for (int j = i; j > 0 && t[j] < t[j - 1]; j--) { float x = t[j]; t[j] = t[j - 1]; t[j - 1] = x; }
    return t[reps / 2];
}
