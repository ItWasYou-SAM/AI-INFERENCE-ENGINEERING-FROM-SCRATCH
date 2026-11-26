# Module 6 labs: GEMM and memory optimization 🔴

Plain CUDA C++, like the [Module 4](../m04/README.md) and [Module 5](../m05/README.md) labs. Each file builds on its own, checks its
answers and prints PASS or FAIL. `common.cuh` adds matrix setup, an error check and a cuBLAS reference to the earlier helpers.

The matmul labs compare against cuBLAS, NVIDIA's own GEMM library, which ships with the CUDA toolkit:

```bash
cd labs/m06/6.1
nvcc -O2 -arch=native -DUSE_CUBLAS matmul.cu -o matmul -lcublas && ./matmul
```

Without cuBLAS, drop `-DUSE_CUBLAS -lcublas`: the kernels are then checked against each other instead (6.7 needs cuBLAS).
On Colab, see the [Module 4 README](../m04/README.md). If your host compiler is newer than your CUDA release supports, add
`-allow-unsupported-compiler`.

| Lab | File | Build flags | What it shows |
|:--|:--|:--|:--|
| 6.1 | `6.1/matmul.cu` | `-DUSE_CUBLAS -lcublas` | naive, coalesced and shared-memory tiled matmul vs cuBLAS |
| 6.2 | `6.2/register_tiling.cu` | `-DUSE_CUBLAS -lcublas` | 1D and 2D register tiling, then `float4`: from 15% to 94% of cuBLAS |
| 6.3 | `6.3/register_pressure.cu` | `-DUSE_CUBLAS -lcublas` | per-thread tile size and `__launch_bounds__` vs registers, spills, occupancy and speed |
| 6.4 | `6.4/bank_conflicts.cu` | | shared-memory reads at strides 0–33, against the predicted bank conflicts |
| 6.5 | `6.5/waves.cu` | | the wave staircase, with a synthetic kernel and with a real matmul |
| 6.6 | `6.6/ilp.cu` | | FMA chains and loads in flight: ILP vs occupancy |
| 6.7 | `6.7/tensor_cores.cu` | `-DUSE_CUBLAS -lcublas` | FP32, TF32, FP16, BF16 on tensor cores; two WMMA kernels; errors |
| 6.8 | `6.8/kernels.cu`, `6.8/inspect.sh` | | PTX and SASS of small kernel pairs; run `./inspect.sh all counts` |

The numbers in the chapters were measured on an RTX 3050 6GB Laptop GPU (sm_86, 20 SMs, ~168 GB/s, cuBLAS SGEMM ≈ 4.5 TFLOP/s),
on AC power with the memory at full clock, built with CUDA 13.5 and cuBLAS 13.10. Yours will differ; compare the ratios.
