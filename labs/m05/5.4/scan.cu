// Lab 5.4 — Prefix sum (scan)
//
// An inclusive scan of 16 M ints, built in three phases:
//   1. scan each tile independently, and write each tile's total to `sums`
//   2. scan `sums` (recursively, with the same code)
//   3. add the scanned total of all earlier tiles to every element of a tile
//
// Phase 1 is done with four different block-level scans:
//   hillis-steele  one element per thread; log2(T) steps, each adding at a distance (n log n adds)
//   blelloch       two elements per thread; up-sweep then down-sweep (about 2n adds)
//   warp shuffle   one element per thread; __shfl_up_sync inside each warp, then across warps
//   4 per thread   each thread scans 4 elements in registers, then a warp-shuffle scan of thread totals
//
// Last, "reduce-then-scan" reorganizes the phases so the data crosses memory 3 times, not 4.
//
// Then it uses the scan for stream compaction: keep only the elements above a threshold.
//
// Build & run:  nvcc -O2 -arch=native scan.cu -o scan && ./scan
#include <random>
#include <vector>
#include "../common.cuh"

constexpr unsigned FULL = 0xffffffffu;

// ---- Hillis–Steele: TILE = 512, 512 threads -----------------------------------------------------
__global__ void scan_hillis_steele(const int* in, int* out, int* sums, int n) {
    constexpr int T = 512;
    __shared__ int buf[2][T];                   // double buffer: read one, write the other
    int t = threadIdx.x, i = blockIdx.x * T + t;
    int pin = 0;
    buf[0][t] = i < n ? in[i] : 0;
    __syncthreads();
    for (int off = 1; off < T; off *= 2) {
        int pout = 1 - pin;
        buf[pout][t] = buf[pin][t] + (t >= off ? buf[pin][t - off] : 0);
        __syncthreads();
        pin = pout;
    }
    if (i < n) out[i] = buf[pin][t];
    if (t == T - 1) sums[blockIdx.x] = buf[pin][t];
}

// ---- Blelloch: TILE = 512, 256 threads, two elements each ---------------------------------------
__global__ void scan_blelloch(const int* in, int* out, int* sums, int n) {
    constexpr int T = 512;
    __shared__ int s[T];
    int t = threadIdx.x, base = blockIdx.x * T;
    int a = base + t < n ? in[base + t] : 0;
    int b = base + t + T / 2 < n ? in[base + t + T / 2] : 0;
    s[t] = a; s[t + T / 2] = b;
    // up-sweep: build partial sums in place, like a reduction tree
    int off = 1;
    for (int d = T / 2; d > 0; d /= 2) {
        __syncthreads();
        if (t < d) {
            int ai = off * (2 * t + 1) - 1, bi = off * (2 * t + 2) - 1;
            s[bi] += s[ai];
        }
        off *= 2;
    }
    // down-sweep: clear the root, then push prefixes back down the tree
    if (t == 0) { if (sums) sums[blockIdx.x] = s[T - 1]; s[T - 1] = 0; }
    for (int d = 1; d < T; d *= 2) {
        off /= 2;
        __syncthreads();
        if (t < d) {
            int ai = off * (2 * t + 1) - 1, bi = off * (2 * t + 2) - 1;
            int x = s[ai];
            s[ai] = s[bi];
            s[bi] += x;
        }
    }
    __syncthreads();
    // s now holds the exclusive scan; add the element itself to make it inclusive
    if (base + t < n) out[base + t] = s[t] + a;
    if (base + t + T / 2 < n) out[base + t + T / 2] = s[t + T / 2] + b;
}

// ---- Warp shuffle: TILE = 512, 512 threads -------------------------------------------------------
__device__ int warp_inclusive_scan(int v) {
    int lane = threadIdx.x % 32;
    for (int off = 1; off < 32; off *= 2) {
        int up = __shfl_up_sync(FULL, v, off);   // value from lane - off
        if (lane >= off) v += up;
    }
    return v;
}

// Inclusive scan of one value per thread across a block of NT threads.
template <int NT>
__device__ int block_inclusive_scan(int v, int* block_total) {
    __shared__ int warp_tot[NT / 32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_inclusive_scan(v);
    if (lane == 31) warp_tot[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = lane < NT / 32 ? warp_tot[lane] : 0;
        w = warp_inclusive_scan(w);
        if (lane < NT / 32) warp_tot[lane] = w;
    }
    __syncthreads();
    if (warp > 0) v += warp_tot[warp - 1];
    *block_total = warp_tot[NT / 32 - 1];
    return v;
}

__global__ void scan_warp_shuffle(const int* in, int* out, int* sums, int n) {
    constexpr int T = 512;
    int i = blockIdx.x * T + threadIdx.x;
    int total;
    int v = block_inclusive_scan<T>(i < n ? in[i] : 0, &total);
    if (i < n) out[i] = v;
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

// ---- 4 elements per thread: TILE = 2048, 512 threads ---------------------------------------------
__global__ void scan_4_per_thread(const int* in, int* out, int* sums, int n) {
    constexpr int NT = 512, T = 4 * NT;
    int i = blockIdx.x * T + 4 * threadIdx.x;
    int4 q = {0, 0, 0, 0};
    if (i + 3 < n) q = *reinterpret_cast<const int4*>(in + i);
    else { if (i < n) q.x = in[i]; if (i + 1 < n) q.y = in[i + 1]; if (i + 2 < n) q.z = in[i + 2]; }
    q.y += q.x; q.z += q.y; q.w += q.z;          // serial scan of my 4, in registers
    int total;
    int incl = block_inclusive_scan<NT>(q.w, &total);
    int before = incl - q.w;                     // sum of all threads before me
    q.x += before; q.y += before; q.z += before; q.w += before;
    if (i + 3 < n) *reinterpret_cast<int4*>(out + i) = q;
    else { if (i < n) out[i] = q.x; if (i + 1 < n) out[i + 1] = q.y; if (i + 2 < n) out[i + 2] = q.z; }
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

// ---- Phase 3 ------------------------------------------------------------------------------------
__global__ void add_tile_offsets(int* out, const int* scanned_sums, int n, int tile) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = i / tile;
    if (i < n && b > 0) out[i] += scanned_sums[b - 1];
}

// ---- The recursive driver -----------------------------------------------------------------------
struct Scanner {
    void (*kernel)(const int*, int*, int*, int);
    int tile, threads;
    std::vector<int*> levels;                   // one `sums` buffer per recursion level

    Scanner(void (*k)(const int*, int*, int*, int), int tile, int threads, int n) : kernel(k), tile(tile), threads(threads) {
        for (int m = n; m > tile; m = (m + tile - 1) / tile) {
            int* p;
            CUDA_CHECK(cudaMalloc(&p, ((m + tile - 1) / tile) * sizeof(int)));
            levels.push_back(p);
        }
        int* p;                                  // the last level still writes one total
        CUDA_CHECK(cudaMalloc(&p, sizeof(int)));
        levels.push_back(p);
    }
    ~Scanner() { for (auto p : levels) cudaFree(p); }

    void run(const int* in, int* out, int n, int level = 0) {
        int blocks = (n + tile - 1) / tile;
        kernel<<<blocks, threads>>>(in, out, levels[level], n);              // phase 1
        if (blocks == 1) return;
        run(levels[level], levels[level], blocks, level + 1);              // phase 2, in place
        add_tile_offsets<<<(n + 255) / 256, 256>>>(out, levels[level], n, tile);   // phase 3
    }
};

// ---- Reduce-then-scan: 3 trips through memory instead of 4 ---------------------------------------
// Pass 1 only *reads* each tile and writes its total. Pass 2 scans the totals.
// Pass 3 reads the tile again and writes the final values, offset included.
// The output is written once, instead of written and then re-read and rewritten.
__global__ void tile_totals(const int* in, int* sums, int n) {
    constexpr int NT = 512, T = 4 * NT;
    int i = blockIdx.x * T + 4 * threadIdx.x;
    int v = 0;
    if (i + 3 < n) { int4 q = *reinterpret_cast<const int4*>(in + i); v = q.x + q.y + q.z + q.w; }
    else for (int j = i; j < n; j++) v += in[j];
    int total;
    block_inclusive_scan<NT>(v, &total);        // (a reduction would do; reuse the scan)
    if (threadIdx.x == 0) sums[blockIdx.x] = total;
}

__global__ void scan_with_offset(const int* in, int* out, const int* scanned_sums, int n) {
    constexpr int NT = 512, T = 4 * NT;
    int i = blockIdx.x * T + 4 * threadIdx.x;
    int4 q = {0, 0, 0, 0};
    if (i + 3 < n) q = *reinterpret_cast<const int4*>(in + i);
    else { if (i < n) q.x = in[i]; if (i + 1 < n) q.y = in[i + 1]; if (i + 2 < n) q.z = in[i + 2]; }
    q.y += q.x; q.z += q.y; q.w += q.z;
    int total;
    int before = block_inclusive_scan<NT>(q.w, &total) - q.w;
    if (blockIdx.x > 0) before += scanned_sums[blockIdx.x - 1];
    q.x += before; q.y += before; q.z += before; q.w += before;
    if (i + 3 < n) *reinterpret_cast<int4*>(out + i) = q;
    else { if (i < n) out[i] = q.x; if (i + 1 < n) out[i + 1] = q.y; if (i + 2 < n) out[i + 2] = q.z; }
}

// ---- Stream compaction --------------------------------------------------------------------------
__global__ void make_flags(const int* x, int* flags, int n, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = x[i] > threshold;
}
__global__ void scatter(const int* x, const int* flags, const int* pos, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i]) out[pos[i] - 1] = x[i];   // inclusive scan: my slot is pos - 1
}

int main() {
    print_device();
    const int n = 1 << 24;
    std::mt19937 rng(7);
    std::vector<int> h(n), expect(n), got(n);
    for (auto& v : h) v = rng() % 4;
    for (int i = 0, run = 0; i < n; i++) expect[i] = run += h[i];

    int *in, *out;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    std::printf("\nInclusive scan of %d ints (%.0f MB in, %.0f MB out):\n", n, n * 4 / 1e6, n * 4 / 1e6);
    std::printf("  %-16s %6s %9s %10s\n", "phase 1 kernel", "tile", "time", "GB/s");
    struct V { const char* name; void (*k)(const int*, int*, int*, int); int tile, threads; };
    V vs[] = {{"hillis-steele", scan_hillis_steele, 512, 512},
              {"blelloch", scan_blelloch, 512, 256},
              {"warp shuffle", scan_warp_shuffle, 512, 512},
              {"4 per thread", scan_4_per_thread, 2048, 512}};
    for (auto& v : vs) {
        Scanner s(v.k, v.tile, v.threads, n);
        float ms = time_ms([&] { s.run(in, out, n); });
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(got.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = got == expect;
        std::printf("  %-16s %6d %6.3f ms %10.1f  %s\n", v.name, v.tile, ms, 2.0 * n * 4 / ms / 1e6, ok ? "PASS" : "FAIL");
    }
    {
        const int T = 2048, blocks = (n + T - 1) / T;
        int* sums;
        CUDA_CHECK(cudaMalloc(&sums, blocks * sizeof(int)));
        Scanner small(scan_4_per_thread, T, 512, blocks);
        float ms = time_ms([&] {
            tile_totals<<<blocks, 512>>>(in, sums, n);
            small.run(sums, sums, blocks);
            scan_with_offset<<<blocks, 512>>>(in, out, sums, n);
        });
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaMemcpy(got.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("  %-16s %6d %6.3f ms %10.1f  %s\n", "reduce-then-scan", T, ms, 2.0 * n * 4 / ms / 1e6, got == expect ? "PASS" : "FAIL");
        cudaFree(sums);
    }
    float ms = time_ms([&] { cudaMemcpy(out, in, n * sizeof(int), cudaMemcpyDeviceToDevice); });
    std::printf("  %-16s %6s %6.3f ms %10.1f  (the floor: read once, write once)\n", "cudaMemcpy D2D", "", ms, 2.0 * n * 4 / ms / 1e6);

    // Stream compaction: keep the values > 2 (about a quarter of them).
    int *flags, *pos, *kept;
    CUDA_CHECK(cudaMalloc(&flags, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pos, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&kept, n * sizeof(int)));
    Scanner s(scan_4_per_thread, 2048, 512, n);
    ms = time_ms([&] {
        make_flags<<<n / 256, 256>>>(in, flags, n, 2);
        s.run(flags, pos, n);
        scatter<<<n / 256, 256>>>(in, flags, pos, kept, n);
    });
    CUDA_CHECK_LAUNCH();
    int count;
    CUDA_CHECK(cudaMemcpy(&count, pos + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int expect_count = 0;
    for (int v : h) expect_count += v > 2;
    std::vector<int> k(count);
    CUDA_CHECK(cudaMemcpy(k.data(), kept, count * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = count == expect_count;
    for (int v : k) ok &= v == 3;
    std::printf("\nCompaction (keep x > 2): kept %d of %d in %.3f ms  %s\n", count, n, ms, ok ? "PASS" : "FAIL");

    cudaFree(in); cudaFree(out); cudaFree(flags); cudaFree(pos); cudaFree(kept);
}
