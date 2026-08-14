"""Small runtime helpers shared by the H200 entry points.

Keep this module stdlib-only: ``sample.py`` imports and runs it before importing
PyTorch so Triton's compiler subprocess can find the NVIDIA driver library.
"""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Optional


EXPECTED_H200_PACKAGES = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "triton": "3.4.0",
}


def _prepend_env_path(name: str, directory: str) -> None:
    entries = [entry for entry in os.environ.get(name, "").split(os.pathsep) if entry]
    if directory not in entries:
        os.environ[name] = os.pathsep.join([directory, *entries])


def _libcuda_candidates() -> list[Path]:
    candidates: list[Path] = []

    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        result = None

    if result is not None:
        for line in result.stdout.splitlines():
            if "libcuda.so.1" in line and "=>" in line:
                candidates.append(Path(line.rsplit("=>", 1)[1].strip()))

    candidates.extend(
        Path(path)
        for path in (
            "/usr/local/nvidia/lib64/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/local/cuda/compat/libcuda.so.1",
        )
    )
    return candidates


def _libcuda_linker_directory(libcuda_path: Path) -> Path:
    """Return a directory where GCC's ``-lcuda`` can resolve ``libcuda.so``.

    Runtime-only NVIDIA container mounts sometimes expose ``libcuda.so.1`` but
    omit the unversioned development symlink required by the linker. In that
    case, create a per-user alias under the system temporary directory without
    modifying the driver mount.
    """

    driver_directory = libcuda_path.parent
    if (driver_directory / "libcuda.so").is_file():
        return driver_directory

    alias_directory = (
        Path(tempfile.gettempdir()) / f"reward-guidance-libcuda-{os.getuid()}"
    )
    alias_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    alias_path = alias_directory / "libcuda.so"
    if alias_path.is_symlink():
        if alias_path.resolve() != libcuda_path.resolve():
            alias_path.unlink()
    elif alias_path.exists():
        raise RuntimeError(
            f"Cannot create the Triton libcuda linker alias: {alias_path} "
            "already exists and is not a symlink."
        )
    if not alias_path.exists():
        alias_path.symlink_to(libcuda_path.resolve())
    return alias_directory


def configure_linux_libcuda_path() -> Optional[str]:
    """Expose the host driver directory to Triton's GCC subprocess.

    NVIDIA containers commonly mount ``libcuda.so.1`` under
    ``/usr/local/nvidia/lib64``. Triton invokes GCC in a subprocess, so both
    the compiler search path and the child dynamic-loader path must include
    that directory.
    """

    if sys.platform != "linux":
        return None

    for candidate in _libcuda_candidates():
        if candidate.is_file():
            directory = str(candidate.parent)
            linker_directory = str(_libcuda_linker_directory(candidate))
            # Triton 3.4 reads this before compiling cuda_utils.c and passes the
            # directory directly to GCC as `-L... -lcuda`.
            os.environ["TRITON_LIBCUDA_PATH"] = linker_directory
            _prepend_env_path("LIBRARY_PATH", linker_directory)
            _prepend_env_path("LD_LIBRARY_PATH", linker_directory)
            _prepend_env_path("LD_LIBRARY_PATH", directory)
            return str(candidate)
    return None


def validate_h200_runtime(torch_module: Any) -> bool:
    """Reject an untested Python/PyTorch stack when the active GPU is an H200.

    Returns ``True`` on H200 and ``False`` on other platforms/devices. This is
    deliberately lightweight compared with ``h200_preflight.py`` so every
    direct ``sample.py`` invocation gets the same dependency guard.
    """

    if sys.platform != "linux" or not torch_module.cuda.is_available():
        return False

    device = torch_module.cuda.current_device()
    if "H200" not in torch_module.cuda.get_device_name(device).upper():
        return False

    problems = []
    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python 3.11 required, found {sys.version.split()[0]}")

    for package, expected in EXPECTED_H200_PACKAGES.items():
        try:
            actual = metadata.version(package).split("+", 1)[0]
        except metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            problems.append(f"{package}=={expected} required, found {actual}")

    if torch_module.version.cuda != "12.8":
        problems.append(
            f"PyTorch CUDA 12.8 build required, found {torch_module.version.cuda!r}"
        )

    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            f"Unsupported H200 runtime: {details}. Run `uv sync --frozen` from "
            "the repository root before starting FLUX."
        )
    return True
