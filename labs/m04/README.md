# Module 4 labs: CUDA programming fundamentals 🔴

These labs are plain CUDA C++. Each `.cu` file builds on its own and checks its own answer.

**On your own NVIDIA GPU** (CUDA toolkit installed):

```bash
cd labs/m04/4.1
nvcc -O2 -arch=native vector_add.cu -o vector_add && ./vector_add
```

**On Google Colab** (free T4): Runtime → Change runtime type → T4 GPU, then in a cell:

```bash
!git clone <this repository> course
%cd course/labs/m04/4.1
!nvcc -O2 -arch=native vector_add.cu -o vector_add && ./vector_add
```

If your host compiler is newer than your CUDA release supports, add `-allow-unsupported-compiler`.

| Lab | File | What it shows |
|:--|:--|:--|
| 4.1 | `4.1/vector_add.cu` | launching a kernel, async launches, a checked vector add |
| 4.2 | `4.2/indexing.cu` | global index, grid-stride loops, 2D grids |
| 4.3 | `4.3/transfers.cu` | pageable vs pinned transfer bandwidth, transfer latency |
| 4.4 | `4.4/shared_sum.cu` | shared memory, tree reduction, what a missing `__syncthreads` does |
| 4.5 | `4.5/race.cu` | a data race, atomics, block-level aggregation |
| 4.6 | `4.6/streams.cu` | launch overhead, CUDA Graphs, overlapping copies and kernels with streams |
| 4.7 | `4.7/coalescing.cu` | strided access, coalescing, a tiled transpose |

The numbers quoted in the chapters were measured on an RTX 3050 6GB Laptop GPU (sm_86, ~168 GB/s).
Yours will differ; the ratios are what to compare.
