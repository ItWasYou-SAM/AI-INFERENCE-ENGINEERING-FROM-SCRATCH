"""
Lab 0.3 — What can this machine run?

Checks Python, NumPy, PyTorch, CUDA, the GPU, nvcc, Triton and vLLM, then
tells you which lab tiers (🟢 CPU · 🟡 GPU optional · 🔴 GPU required) you can run here.

Run:  python labs/m00/0.3/check_env.py
"""
import importlib
import platform
import shutil
import subprocess
import sys

OK, NO, MAYBE = "✔", "✘", "–"


def has(mod: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def row(label: str, status: str, detail: str = ""):
    print(f"  {status} {label:<22} {detail}")


def main():
    print(f"\nPython {platform.python_version()} on {platform.system()} {platform.machine()}\n")
    py_ok = sys.version_info >= (3, 10)
    row("Python ≥ 3.10", OK if py_ok else NO, "" if py_ok else "upgrade Python")

    np_v = has("numpy")
    row("numpy", OK if np_v else NO, np_v or "pip install numpy")

    torch_v = has("torch")
    row("torch", OK if torch_v else MAYBE, torch_v or "optional: pip install torch")

    cuda = False
    if torch_v:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            p = torch.cuda.get_device_properties(0)
            row("CUDA GPU", OK, f"{p.name} · {p.total_memory / 1e9:.0f} GB · sm_{p.major}{p.minor} · x{torch.cuda.device_count()}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            row("CUDA GPU", MAYBE, "Apple MPS found: fine for 🟡 labs, not for CUDA/Triton")
        else:
            row("CUDA GPU", NO, "no CUDA device visible to torch")
    elif shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        row("CUDA GPU", MAYBE, f"{out} (install torch to use it)")
        cuda = bool(out)
    else:
        row("CUDA GPU", NO, "none found")

    nvcc = shutil.which("nvcc")
    if nvcc:
        ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        row("nvcc (CUDA C++)", OK, ver)
    else:
        row("nvcc (CUDA C++)", MAYBE, "needed for .cu labs in M4–M6")

    tr = has("triton")
    row("triton", OK if tr else MAYBE, tr or "needed for M8 Triton labs")
    vl = has("vllm")
    row("vllm", OK if vl else MAYBE, vl or "needed for serving labs in M10–M13")

    print("\nYou can run:")
    print(f"  {'🟢 CPU labs':<22} {'yes' if (py_ok and np_v) else 'after: pip install numpy'}")
    print(f"  {'🟡 GPU-optional labs':<22} {'yes (CPU fallback)' if np_v else 'after installing numpy'}"
          + (" — and fast on your GPU" if cuda else ""))
    if cuda:
        todo = [x for x, ok in (("PyTorch", torch_v), ("the CUDA toolkit (nvcc, for .cu labs)", nvcc)) if not ok]
        print(f"  {'🔴 GPU-required labs':<22} yes" + (f", after installing {' and '.join(todo)}" if todo else ""))
    else:
        print(f"  {'🔴 GPU-required labs':<22} use Google Colab / Kaggle (free T4) or rent a GPU by the hour")
    print()


if __name__ == "__main__":
    main()
