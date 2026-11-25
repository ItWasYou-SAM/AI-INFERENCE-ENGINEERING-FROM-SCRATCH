// Lab 4.2 — Thread, block and grid indexing
//
// 1. Print how (blockIdx, threadIdx) map to a global index for a small 1D launch.
// 2. A grid-stride loop: a fixed-size grid that handles any n.
// 3. 2D indexing: add two matrices with a 2D grid of 2D blocks.
//
// Build & run:  nvcc -O2 -arch=native indexing.cu -o indexing && ./indexing
#include <vector>
#include "../common.cuh"

__global__ void show_index(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = blockIdx.x * 1000 + threadIdx.x;   // encode who wrote it
}

// Grid-stride loop: thread i handles i, i + stride, i + 2*stride, ...
__global__ void scale(float* x, float s, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        x[i] *= s;
}

// Row-major matrices: element (row, col) lives at row * cols + col.
__global__ void mat_add(const float* a, const float* b, float* c, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> columns (contiguous)
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y -> rows
    if (row < rows && col < cols) {
        int i = row * cols + col;
        c[i] = a[i] + b[i];
    }
}

int main() {
    print_device();

    // 1) 3 blocks of 4 threads -> 12 global indices
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, 12 * sizeof(int)));
    show_index<<<3, 4>>>(d_out);
    CUDA_CHECK_LAUNCH();
    int h_out[12];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost));
    std::printf("\n1) i = blockIdx.x * blockDim.x + threadIdx.x  (3 blocks x 4 threads)\n");
    for (int i = 0; i < 12; i++)
        std::printf("  i=%2d  <- block %d, thread %d\n", i, h_out[i] / 1000, h_out[i] % 1000);
    cudaFree(d_out);

    // 2) grid-stride loop with a deliberately small grid
    const int n = 10'000'000;
    std::vector<float> h(n, 1.0f);
    float* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    scale<<<sms * 4, 256>>>(d, 3.0f, n);     // far fewer threads than elements
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (float v : h) bad += v != 3.0f;
    std::printf("\n2) grid-stride: %d threads covered %d elements: %s\n", sms * 4 * 256, n, bad ? "FAIL" : "PASS");
    cudaFree(d);

    // 3) 2D: a 3000 x 5000 matrix with 32 x 8 blocks
    const int rows = 3000, cols = 5000, m = rows * cols;
    std::vector<float> ha(m), hb(m), hc(m);
    for (int i = 0; i < m; i++) { ha[i] = float(i % 97); hb[i] = float(i % 13); }
    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, m * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(a, ha.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, hb.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    dim3 block(32, 8);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    mat_add<<<grid, block>>>(a, b, c, rows, cols);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(hc.data(), c, m * sizeof(float), cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < m; i++) bad += hc[i] != ha[i] + hb[i];
    std::printf("\n3) 2D: grid (%u, %u) of blocks (%u, %u) = %u threads for %d elements: %s\n",
                grid.x, grid.y, block.x, block.y, grid.x * grid.y * block.x * block.y, m, bad ? "FAIL" : "PASS");
    cudaFree(a); cudaFree(b); cudaFree(c);
    return bad != 0;
}
