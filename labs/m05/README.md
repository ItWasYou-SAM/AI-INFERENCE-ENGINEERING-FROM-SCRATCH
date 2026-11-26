# Module 5 labs: parallel patterns 🔴

Plain CUDA C++, like the [Module 4 labs](../m04/README.md). Each `.cu` file builds on its own, checks its own answers and prints
PASS or FAIL next to every timing. `common.cuh` adds a median-of-N timer to the Module 4 helpers.

```bash
cd labs/m05/5.1
nvcc -O2 -arch=native reduction.cu -o reduction && ./reduction
```

On Colab, see the [Module 4 README](../m04/README.md). If your host compiler is newer than your CUDA release supports, add
`-allow-unsupported-compiler`.

| Lab | File | What it shows |
|:--|:--|:--|
| 5.1 | `5.1/reduction.cu` | five reduction kernels, from divergent tree to grid-stride + `float4` |
| 5.2 | `5.2/shuffles.cu` | `__shfl_down_sync` vs `__shfl_xor_sync`, shuffle-based block sums, one-pass sum with an atomic |
| 5.3 | `5.3/atomics.cu` | atomic contention vs number of addresses, a CAS-loop float max, the deterministic last-block pattern |
| 5.4 | `5.4/scan.cu` | Hillis–Steele, Blelloch and shuffle scans; three-phase vs reduce-then-scan; stream compaction |
| 5.5 | `5.5/histogram.cu` | global atomics vs shared-memory privatization, on uniform, skewed and constant data |
| 5.6 | `5.6/convolution.cu` | 2D convolution: naive, `__constant__` filter, shared-memory tile with halo; causal depthwise conv1d |
| 5.7 | `5.7/softmax.cu` | softmax as 3 kernels, fused, online, in registers, and one warp per row, on three row shapes |

The numbers quoted in the chapters were measured on an RTX 3050 6GB Laptop GPU (sm_86, 20 SMs, ~168 GB/s peak), built with
CUDA 13.5. During these runs the GPU's memory was below its top clock, so bandwidth-bound kernels topped out near 125 GB/s; rerun
at full clock, the 5.1 v5 kernel reaches 162 GB/s. The comparisons between kernels were all measured in the same state. Laptop GPUs
change clocks with power source and temperature, so expect noise between runs, and more on the slowest kernels. Yours will differ;
compare the ratios.
