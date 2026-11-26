// Lab 5.6 — Convolution and stencils
//
// A 2D convolution of a 4096 x 4096 float image with a K x K filter
// (zero padding, same-size output), for K = 3, 7 and 11, three ways:
//
//   naive      filter in global memory; every input pixel is re-read from
//              memory (well, from cache) by each of the K*K outputs that use it
//   constant   filter in __constant__ memory: every thread reads the same weight
//              at the same time, which the constant cache broadcasts
//   tiled      each block loads its input tile plus a (K-1)-wide halo into shared
//              memory once, then computes 32 x 32 outputs from it
//
// Then a 1D causal depthwise convolution, the kind found in Mamba-style
// models: each channel has its own short filter over the time axis.
//
// Build & run:  nvcc -O2 -arch=native convolution.cu -o convolution && ./convolution
#include <cmath>
#include <random>
#include <vector>
#include "../common.cuh"

constexpr int W = 4096, H = 4096, MAXK = 11;
__constant__ float c_filter[MAXK * MAXK];

template <int K>
__global__ void conv_naive(const float* in, const float* filter, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int dy = 0; dy < K; dy++)
        for (int dx = 0; dx < K; dx++) {
            int ix = x + dx - K / 2, iy = y + dy - K / 2;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) acc += filter[dy * K + dx] * in[iy * W + ix];
        }
    out[y * W + x] = acc;
}

template <int K>
__global__ void conv_constant(const float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int dy = 0; dy < K; dy++)
        for (int dx = 0; dx < K; dx++) {
            int ix = x + dx - K / 2, iy = y + dy - K / 2;
            if (ix >= 0 && ix < W && iy >= 0 && iy < H) acc += c_filter[dy * K + dx] * in[iy * W + ix];
        }
    out[y * W + x] = acc;
}

// 32 x 8 threads compute a 32 x 32 output tile: each thread does 4 rows.
template <int K>
__global__ void conv_tiled(const float* in, float* out) {
    constexpr int T = 32, R = K / 2, S = T + K - 1;      // S: tile plus halo on both sides
    __shared__ float tile[S][S + 1];                      // +1: avoid bank conflicts (6.4)
    int x0 = blockIdx.x * T, y0 = blockIdx.y * T;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int j = tid; j < S * S; j += blockDim.x * blockDim.y) {   // cooperative load, halo included
        int sy = j / S, sx = j % S;
        int ix = x0 + sx - R, iy = y0 + sy - R;
        tile[sy][sx] = (ix >= 0 && ix < W && iy >= 0 && iy < H) ? in[iy * W + ix] : 0.0f;
    }
    __syncthreads();
    int tx = threadIdx.x;
    for (int ty = threadIdx.y; ty < T; ty += blockDim.y) {
        float acc = 0.0f;
        #pragma unroll
        for (int dy = 0; dy < K; dy++)
            #pragma unroll
            for (int dx = 0; dx < K; dx++) acc += c_filter[dy * K + dx] * tile[ty + dy][tx + dx];
        out[(y0 + ty) * W + x0 + tx] = acc;
    }
}

// Causal depthwise conv1d: x is [T, C] (time-major, channels contiguous),
// y[t][c] = sum_k w[c][k] * x[t - (K-1) + k][c]. One thread per (t, c).
template <int K>
__global__ void causal_conv1d(const float* x, const float* w, float* y, int T, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        int ts = t - (K - 1) + k;
        if (ts >= 0) acc += w[c * K + k] * x[ts * C + c];
    }
    y[t * C + c] = acc;
}

template <int K>
void run_2d(const float* d_in, float* d_out, float* d_filter, const std::vector<float>& img, std::mt19937& rng) {
    std::vector<float> f(K * K);
    std::uniform_real_distribution<float> U(-1, 1);
    for (auto& v : f) v = U(rng);
    CUDA_CHECK(cudaMemcpy(d_filter, f.data(), K * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, f.data(), K * K * sizeof(float)));

    // CPU reference on a sample of pixels
    std::vector<std::pair<int, int>> pts;
    std::vector<float> ref;
    for (int s = 0; s < 4000; s++) {
        int x = rng() % W, y = rng() % H;
        if (s < 4) { x = s % 2 ? W - 1 : 0; y = s / 2 ? H - 1 : 0; }   // include the corners
        double acc = 0;
        for (int dy = 0; dy < K; dy++)
            for (int dx = 0; dx < K; dx++) {
                int ix = x + dx - K / 2, iy = y + dy - K / 2;
                if (ix >= 0 && ix < W && iy >= 0 && iy < H) acc += f[dy * K + dx] * img[iy * W + ix];
            }
        pts.push_back({x, y});
        ref.push_back(float(acc));
    }
    std::vector<float> got(size_t(W) * H);
    auto check = [&] {
        CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < pts.size(); i++)
            if (std::fabs(got[pts[i].second * W + pts[i].first] - ref[i]) > 1e-3f) return false;
        return true;
    };

    dim3 b16(16, 16), g16(W / 16, H / 16), b32x8(32, 8), g32(W / 32, H / 32);
    float ms[3];
    bool ok[3];
    ms[0] = time_ms([&] { conv_naive<K><<<g16, b16>>>(d_in, d_filter, d_out); }, 10); ok[0] = check();
    ms[1] = time_ms([&] { conv_constant<K><<<g16, b16>>>(d_in, d_out); }, 10);        ok[1] = check();
    ms[2] = time_ms([&] { conv_tiled<K><<<g32, b32x8>>>(d_in, d_out); }, 10);         ok[2] = check();
    CUDA_CHECK_LAUNCH();
    double flops = 2.0 * K * K * W * H;
    std::printf("  %2dx%-2d", K, K);
    for (int i = 0; i < 3; i++) std::printf("   %7.3f ms %6.0f GFLOP/s%s", ms[i], flops / ms[i] / 1e6, ok[i] ? "" : " FAIL");
    std::printf("\n");
}

int main() {
    print_device();
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> U(0, 1);
    std::vector<float> img(size_t(W) * H);
    for (auto& v : img) v = U(rng);
    float *d_in, *d_out, *d_filter;
    CUDA_CHECK(cudaMalloc(&d_in, img.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, img.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, MAXK * MAXK * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::printf("\n2D convolution, %d x %d floats (%.0f MB in, %.0f MB out):\n", W, H, W * H * 4 / 1e6, W * H * 4 / 1e6);
    std::printf("  %-5s   %-27s   %-27s   %-27s\n", "K", "naive", "constant", "tiled");
    run_2d<3>(d_in, d_out, d_filter, img, rng);
    run_2d<7>(d_in, d_out, d_filter, img, rng);
    run_2d<11>(d_in, d_out, d_filter, img, rng);
    float ms = time_ms([&] { cudaMemcpy(d_out, d_in, img.size() * sizeof(float), cudaMemcpyDeviceToDevice); });
    std::printf("  cudaMemcpy of the image: %.3f ms (the floor for any K)\n", ms);

    // Causal depthwise conv1d: 8192 time steps x 2048 channels, K = 4
    const int T = 8192, C = 2048, K = 4;
    std::vector<float> hx(size_t(T) * C), hw(C * K);
    for (auto& v : hx) v = U(rng);
    for (auto& v : hw) v = U(rng);
    float *x, *w, *y;
    CUDA_CHECK(cudaMalloc(&x, hx.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w, hw.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, hx.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));
    ms = time_ms([&] { causal_conv1d<K><<<dim3(C / 256, T), 256>>>(x, w, y, T, C); });
    CUDA_CHECK_LAUNCH();
    std::vector<float> hy(hx.size());
    CUDA_CHECK(cudaMemcpy(hy.data(), y, hy.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int s = 0; s < 2000; s++) {
        int t = s < 3 ? s : rng() % T, c = rng() % C;        // include t = 0, 1, 2 (partial windows)
        double acc = 0;
        for (int k = 0; k < K; k++) { int ts = t - (K - 1) + k; if (ts >= 0) acc += hw[c * K + k] * hx[size_t(ts) * C + c]; }
        ok &= std::fabs(hy[size_t(t) * C + c] - acc) < 1e-4;
    }
    std::printf("\nCausal depthwise conv1d, %d steps x %d channels, K = %d: %.3f ms, %.1f GB/s  %s\n",
                T, C, K, ms, 2.0 * T * C * 4 / ms / 1e6, ok ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_filter); cudaFree(x); cudaFree(w); cudaFree(y);
}
