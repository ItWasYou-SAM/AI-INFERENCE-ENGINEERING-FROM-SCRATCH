# Module 7 labs: Profiling 🔴

The CUDA labs build like the [Module 6](../m06/README.md) ones; `common.cuh` adds NVTX ranges to the earlier helpers. The 7.4 lab
is Python and needs PyTorch. All of them are meant to be run under the profilers that ship with the CUDA toolkit:
Nsight Systems (`nsys`) and Nsight Compute (`ncu`). Each chapter lists the exact commands.

| Lab | Files | Build / run | What it shows |
|:--|:--|:--|:--|
| 7.2 | `7.2/tools.cu`, `step_timeline.py`, `top_sass.py` | `nvcc -O2 -arch=native -lineinfo tools.cu -o tools && ./tools` | one program under both profilers: a timeline, NVTX ranges, sectors per request, warp stalls, the per-instruction source view, and a wrong DRAM counter |
| 7.4 | `7.4/profile_block.py` | `python profile_block.py` | `torch.profiler` on a small GPT stack: prefill, prefill with a hidden `.item()`, and decode; traces for Perfetto |
| 7.5 | `7.5/mlp.cu`, `clocks.py` | `nvcc -O2 -arch=native -DUSE_CUBLAS mlp.cu -o mlp -lcublas -lcublasLt && ./mlp` | the case study: an FP16 MLP block from 5.9 to 15.9 TFLOP/s in four profiled fixes |

**PyTorch for 7.4.** A CUDA build in a virtual environment (the 7.4 lab also runs on a CPU, with CPU-only tables):

```bash
python -m venv labs/.venv && source labs/.venv/bin/activate
pip install --no-cache-dir torch numpy --index-url https://download.pytorch.org/whl/cu132   # pick the build for your driver
```

**GPU counters.** If `ncu` stops with `ERR_NVGPUCTRPERM`, the driver lets only root read performance counters. Run `ncu` with `sudo`, or
allow it for all users (on a machine you don't share):

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo mkinitcpio -P      # or: sudo update-initramfs -u   (Debian/Ubuntu); then reboot
```

The numbers in the chapters were measured on an RTX 3050 6GB Laptop GPU (sm_86, 20 SMs, ~168 GB/s), on AC power with the laptop in its
**performance** power profile, with CUDA 13.5, cuBLAS 13.10, Nsight Systems 2026.5, Nsight Compute 2026.4 and PyTorch 2.15 (CUDA 13.2
build). The power profile matters: in power-saver mode the CPU-bound decode step in 7.4 took more than twice as long. Yours will differ;
compare the ratios, and record your clocks and power profile with your numbers.
