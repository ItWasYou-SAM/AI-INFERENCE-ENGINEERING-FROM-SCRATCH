// Lab 6.8 — Reading PTX and SASS
//
// Small kernels, in pairs, that differ in one detail. Compile them and read what
// the compiler made of each, with ./inspect.sh (see the comments there).
//
// extern "C" keeps the names unmangled, so `cuobjdump -fun saxpy` finds them.
// The main() runs each kernel once and checks it, so the file also builds and runs normally:
//   nvcc -O2 -arch=native kernels.cu -o kernels && ./kernels
#include <cstdio>
#include <cmath>
#include <vector>
#include "../common.cuh"

extern "C" {

// 1. The baseline: one float per thread.
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// 2. The same with 16-byte accesses: one LDG.E.128 instead of four LDG.E.
__global__ void saxpy4(float a, const float4* x, float4* y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 xv = x[i], yv = y[i];
        y[i] = make_float4(a * xv.x + yv.x, a * xv.y + yv.y, a * xv.z + yv.z, a * xv.w + yv.w);
    }
}

// 3. Pointer aliasing. Without __restrict__, the store to out[i] might change w[],
//    so the compiler must reload w[j] after every store.
__global__ void scale_rows(const float* w, float* out, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 4; j++) out[r * 4 + j] = w[0] * w[j] * float(r);
}
__global__ void scale_rows_restrict(const float* __restrict__ w, float* __restrict__ out, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < 4; j++) out[r * 4 + j] = w[0] * w[j] * float(r);
}

// 4. expf vs the hardware approximation __expf (what -use_fast_math selects).
__global__ void exp_accurate(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = expf(x[i]);
}
__global__ void exp_fast(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __expf(x[i]);
}

// 5. A dot product of 8: unrolled, the 8 FMAs need no loop counter or branch.
__global__ void dot8_loop(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    #pragma unroll 1
    for (int k = 0; k < 8; k++) s += a[i * 8 + k] * b[k];
    out[i] = s;
}
__global__ void dot8_unrolled(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) s += a[i * 8 + k] * b[k];
    out[i] = s;
}

}  // extern "C"

int main() {
    print_device();
    const int n = 1 << 20;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; i++) { hx[i] = float(i % 13) * 0.1f; hy[i] = 1.0f; }
    float *x, *y, *z;
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&z, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    saxpy<<<n / 256, 256>>>(2.0f, x, y, n);
    saxpy4<<<n / 4 / 256, 256>>>(2.0f, reinterpret_cast<float4*>(x), reinterpret_cast<float4*>(y), n / 4);
    std::vector<float> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; i++) ok &= std::fabs(got[i] - (1.0f + 4.0f * hx[i])) < 1e-5f;
    std::printf("saxpy then saxpy4: %s\n", ok ? "PASS" : "FAIL");

    // expf vs __expf, on inputs from -80 to 80 (__expf's error grows with |x|)
    const int m = 1601;
    std::vector<float> hv(m);
    for (int i = 0; i < m; i++) hv[i] = -80.0f + 0.1f * i;
    CUDA_CHECK(cudaMemcpy(x, hv.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    exp_accurate<<<(m + 255) / 256, 256>>>(x, y, m);
    exp_fast<<<(m + 255) / 256, 256>>>(x, z, m);
    std::vector<float> e1(m), e2(m);
    CUDA_CHECK(cudaMemcpy(e1.data(), y, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(e2.data(), z, m * sizeof(float), cudaMemcpyDeviceToHost));
    for (float lim : {1.0f, 10.0f, 80.0f}) {
        double w1 = 0, w2 = 0;
        for (int i = 0; i < m; i++) {
            if (std::fabs(hv[i]) > lim) continue;
            double e = std::exp(double(hv[i]));
            w1 = std::fmax(w1, std::fabs(e1[i] - e) / e);
            w2 = std::fmax(w2, std::fabs(e2[i] - e) / e);
        }
        std::printf("|x| <= %4.0f   max relative error: expf %.1e   __expf %.1e\n", lim, w1, w2);
    }
    cudaFree(x); cudaFree(y); cudaFree(z);
}
