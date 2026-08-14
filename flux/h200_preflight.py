"""Fail-fast validation for the CUDA/PyTorch/Triton stack used on H200."""

from __future__ import annotations

import ctypes
from importlib import metadata
import os
import platform
import subprocess
import sys

from h200_runtime import EXPECTED_H200_PACKAGES, configure_linux_libcuda_path


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _nvidia_smi_summary() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"
    return result.stdout.strip()


def main() -> None:
    _require(sys.platform == "linux", "H200 preflight must run on the Linux GPU host.")
    _require(
        sys.version_info[:2] == (3, 11),
        f"Expected Python 3.11, found {platform.python_version()}. Run `uv sync --frozen`.",
    )

    libcuda_path = configure_linux_libcuda_path()
    _require(
        libcuda_path is not None,
        "libcuda.so.1 was not found. The NVIDIA driver must be mounted inside the "
        "container (normally /usr/local/nvidia/lib64 or /usr/lib/x86_64-linux-gnu).",
    )
    try:
        ctypes.CDLL(libcuda_path)
    except OSError as exc:
        raise RuntimeError(f"Found but could not load {libcuda_path}: {exc}") from exc

    import torch

    for package, expected in EXPECTED_H200_PACKAGES.items():
        actual = _base_version(metadata.version(package))
        _require(
            actual == expected,
            f"Expected {package}=={expected}, found {actual}. Run `uv sync --frozen` "
            "from the repository root.",
        )

    _require(torch.cuda.is_available(), "torch.cuda.is_available() is False.")
    _require(
        torch.version.cuda == "12.8",
        f"Expected the CUDA 12.8 PyTorch wheel, found CUDA {torch.version.cuda!r}.",
    )

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    props = torch.cuda.get_device_properties(device_index)
    _require("H200" in props.name.upper(), f"Expected an H200, found {props.name!r}.")
    _require(props.major >= 9, f"Expected compute capability >= 9.0, found {props.major}.{props.minor}.")
    _require(torch.cuda.is_bf16_supported(), "This CUDA runtime does not report BF16 support.")

    print(f"Python:       {platform.python_version()}")
    print(f"PyTorch:      {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"GPU:          {props.name}, sm_{props.major}{props.minor}, {props.total_memory / 2**30:.1f} GiB")
    print(f"Driver:       {_nvidia_smi_summary()}")
    print(f"libcuda.so.1: {libcuda_path}")
    print(f"Triton -L:    {os.environ.get('TRITON_LIBCUDA_PATH')}")

    # Exercise the same BF16 autograd path whose first synchronization point in
    # the failed jobs was torch.autograd.grad.
    x = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16, requires_grad=True)
    loss = (x @ x.transpose(0, 1)).float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device_index)
    del loss, x
    torch.cuda.empty_cache()
    print("CUDA BF16 autograd: PASS")

    # Exercise the attention/checkpoint/autograd combination used by FLUX.
    # Keeping this separate from model loading makes a broken CUDA kernel fail
    # in seconds, before downloading weights or starting a long experiment.
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    def attention(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    q = torch.randn(
        1, 4, 128, 64, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    attended = checkpoint(attention, q, k, v, use_reentrant=False)
    q_grad = torch.autograd.grad(attended.float().square().mean(), q)[0]
    torch.cuda.synchronize(device_index)
    _require(torch.isfinite(q_grad).all().item(), "BF16 SDPA gradient is not finite.")
    del attended, q_grad, q, k, v
    torch.cuda.empty_cache()
    print("CUDA BF16 SDPA checkpoint autograd: PASS")

    # Accessing the active Triton target forces its driver helper to compile and
    # link against libcuda.so.1, reproducing the VLM failure before model load.
    try:
        from triton.runtime import driver

        target = driver.active.get_current_target()
    except Exception as exc:
        raise RuntimeError(
            "Triton could not initialize its CUDA driver helper. Check the "
            "libcuda path printed above and the container's GCC toolchain."
        ) from exc
    print(f"Triton CUDA driver: PASS ({target})")
    print("H200 preflight: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"H200 preflight: FAIL\n{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
