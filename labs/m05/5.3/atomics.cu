// Lab 5.3 — Atomics and compare-and-swap
//
// Part 1: contention. 16 M threads each atomicAdd into one of K counters.
//         How does throughput change as K goes from 1 to 16 M?
// Part 2: atomicCAS. There is no atomicMax for float, so build one from
//         compare-and-swap, count how often the CAS loop retries, and compare
//         with two cheaper tricks.
// Part 3: the "last block" pattern. A one-launch sum whose final additions
//         happen in a fixed order, so the result is bit-identical every run,
//         unlike a float atomicAdd.
//
// Build & run:  nvcc -O2 -arch=native atomics.cu -o atomics && ./atomics
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include "../common.cuh"

constexpr int BLOCK = 256;

// ---- Part 1 ------------------------------------------------------------------------------------
__global__ void count_into(const int* x, unsigned* counters, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&counters[i % k], unsigned(x[i]));
}

// ---- Part 2 ------------------------------------------------------------------------------------
// The general recipe: read the old value, compute the new one, and swap it in
// only if nobody changed the word in between. If somebody did, try again.
__device__ float atomic_max_cas(float* addr, float val, unsigned* retries) {
    unsigned* a = reinterpret_cast<unsigned*>(addr);
    unsigned old = *a, assumed;
    int tries = 0;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;          // nothing to do: don't write at all
        old = atomicCAS(a, assumed, __float_as_uint(val));   // returns what was really there
        tries++;
    } while (old != assumed);                                // someone else got in first: retry
    if (retries && tries > 1) atomicAdd(retries, tries - 1);
    return __uint_as_float(old);
}

template <bool COUNT>
__global__ void max_cas(const float* x, float* out, unsigned* retries, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomic_max_cas(out, x[i], COUNT ? retries : nullptr);
}

// Trick 1: for non-negative floats, the IEEE bit patterns sort in the same
// order as the values, so the integer atomicMax does the job in hardware.
// (Negative floats need a sign flip; see the exercises.)
__global__ void max_int_trick(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMax(reinterpret_cast<int*>(out), __float_as_int(x[i]));
}

// Trick 2: reduce inside the warp first (5.2), so only one lane per warp
// touches global memory.
__global__ void max_warp_then_cas(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? x[i] : 0.0f;
    for (int o = 16; o > 0; o /= 2) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (threadIdx.x % 32 == 0) atomic_max_cas(out, v, nullptr);
}

// ---- Part 3 ------------------------------------------------------------------------------------
__device__ float block_sum(float v) {
    __shared__ float warp_sums[BLOCK / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = lane < BLOCK / 32 ? warp_sums[lane] : 0.0f;
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;                                   // valid in thread 0
}

__device__ float grid_partial(const float* x, int n) {
    float v = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) v += x[i];
    return v;
}

__global__ void sum_atomic(const float* x, float* out, int n) {
    float v = block_sum(grid_partial(x, n));
    if (threadIdx.x == 0) atomicAdd(out, v);    // blocks add in whatever order they finish
}

__global__ void sum_last_block(const float* x, float* partials, unsigned* ticket, float* out, int n) {
    float v = block_sum(grid_partial(x, n));
    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = v;
        __threadfence();                        // make the partial visible to all blocks first
        am_last = atomicAdd(ticket, 1) == gridDim.x - 1;   // take a ticket; the last one sees n-1
    }
    __syncthreads();
    if (am_last) {                              // only the last block to finish gets here
        float t = 0.0f;
        for (int b = threadIdx.x; b < gridDim.x; b += BLOCK) t += partials[b];   // fixed order
        t = block_sum(t);
        if (threadIdx.x == 0) { *out = t; *ticket = 0; }   // reset the ticket for the next launch
    }
}

int main() {
    print_device();
    const int n = 1 << 24;
    std::mt19937 rng(42);

    // Part 1
    std::vector<int> ones(n, 1);
    int* d_ones;
    unsigned* counters;
    CUDA_CHECK(cudaMalloc(&d_ones, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&counters, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d_ones, ones.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    std::printf("\n%d atomicAdds into K counters (thread i adds to counter i %% K):\n", n);
    std::printf("  %10s %10s %16s\n", "K", "time", "atomics / ns");
    for (int k : {1, 2, 32, 1024, 32768, 1 << 20, n}) {
        // timed runs keep adding to the counters; a final clean run checks the answer
        float ms = time_ms([&] { count_into<<<n / BLOCK, BLOCK>>>(d_ones, counters, n, k); });
        CUDA_CHECK(cudaMemset(counters, 0, k * sizeof(unsigned)));
        count_into<<<n / BLOCK, BLOCK>>>(d_ones, counters, n, k);
        CUDA_CHECK_LAUNCH();
        unsigned c0;
        CUDA_CHECK(cudaMemcpy(&c0, counters, sizeof c0, cudaMemcpyDeviceToHost));
        bool ok = c0 == unsigned(n / k);
        std::printf("  %10d %7.3f ms %16.1f  %s\n", k, ms, n / ms / 1e6, ok ? "PASS" : "FAIL");
    }

    // Part 2
    std::vector<float> h(n);
    std::uniform_real_distribution<float> U(0.0f, 1000.0f);
    for (auto& v : h) v = U(rng);
    float expect = *std::max_element(h.begin(), h.end());
    std::vector<float> sorted = h;
    std::sort(sorted.begin(), sorted.end());     // ascending: every value is a new maximum
    float *x, *xs, *out;
    unsigned* retries;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&xs, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&retries, sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(xs, sorted.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("\nMax of %d floats into one address:\n", n);
    auto check = [&](const char* name, float ms, const char* extra) {
        float got;
        CUDA_CHECK(cudaMemcpy(&got, out, sizeof got, cudaMemcpyDeviceToHost));
        std::printf("  %-36s %7.3f ms  %s  %s\n", name, ms, got == expect ? "PASS" : "FAIL", extra);
    };
    for (auto [data, label] : {std::pair{x, "random order"}, std::pair{xs, "ascending order"}}) {
        float ms = time_ms([&] {
            cudaMemsetAsync(out, 0, sizeof(float));
            max_cas<false><<<n / BLOCK, BLOCK>>>(data, out, nullptr, n);
        });
        CUDA_CHECK_LAUNCH();
        // a separate run that also counts the retries
        CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(retries, 0, sizeof(unsigned)));
        max_cas<true><<<n / BLOCK, BLOCK>>>(data, out, retries, n);
        unsigned r;
        CUDA_CHECK(cudaMemcpy(&r, retries, sizeof r, cudaMemcpyDeviceToHost));
        char buf[64];
        std::snprintf(buf, sizeof buf, "CAS retries: %u", r);
        std::string name = std::string("CAS loop, ") + label;
        check(name.c_str(), ms, buf);
    }
    float ms = time_ms([&] {
        cudaMemsetAsync(out, 0, sizeof(float));
        max_int_trick<<<n / BLOCK, BLOCK>>>(xs, out, n);
    });
    CUDA_CHECK_LAUNCH();
    check("integer atomicMax, ascending order", ms, "");
    ms = time_ms([&] {
        cudaMemsetAsync(out, 0, sizeof(float));
        max_warp_then_cas<<<n / BLOCK, BLOCK>>>(xs, out, n);
    });
    CUDA_CHECK_LAUNCH();
    check("warp max, then CAS, ascending order", ms, "");

    // Part 3
    for (auto& v : h) v = U(rng) * 1e-3f;
    CUDA_CHECK(cudaMemcpy(x, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int blocks = sms * 8;
    float* partials;
    unsigned* ticket;
    CUDA_CHECK(cudaMalloc(&partials, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ticket, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(ticket, 0, sizeof(unsigned)));
    std::printf("\nOne-launch float sum of %d values, 6 runs each (bit patterns of the result):\n", n);
    for (int last = 0; last < 2; last++) {
        std::printf("  %-22s", last ? "last-block pattern" : "atomicAdd per block");
        for (int r = 0; r < 6; r++) {
            if (last) sum_last_block<<<blocks, BLOCK>>>(x, partials, ticket, out, n);
            else { CUDA_CHECK(cudaMemset(out, 0, sizeof(float))); sum_atomic<<<blocks, BLOCK>>>(x, out, n); }
            CUDA_CHECK_LAUNCH();
            float got;
            CUDA_CHECK(cudaMemcpy(&got, out, sizeof got, cudaMemcpyDeviceToHost));
            unsigned bits;
            std::memcpy(&bits, &got, 4);
            std::printf(" %08x", bits);
        }
        std::printf("\n");
    }

    cudaFree(d_ones); cudaFree(counters); cudaFree(x); cudaFree(xs); cudaFree(out);
    cudaFree(retries); cudaFree(partials); cudaFree(ticket);
}
