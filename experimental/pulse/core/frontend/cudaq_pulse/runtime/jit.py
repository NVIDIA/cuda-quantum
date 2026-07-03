# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT compiler for lowered MLIR modules targeting cuDensityMat runtime."""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _find_cuda_runtime() -> Optional[Path]:
    """Locate the CUDA runtime library."""
    search_dirs = [
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
        "/usr/local/cuda",
        "/usr/lib/x86_64-linux-gnu",
    ]
    for d in search_dirs:
        if not d:
            continue
        for sub in ("lib64", "lib"):
            candidate = Path(d) / sub / "libcudart.so"
            if candidate.exists():
                return candidate
    return None


def _find_cudm_runtime() -> Optional[Path]:
    """Locate libcudm-runtime.so."""
    env_path = os.environ.get("CUDM_RUNTIME_LIB")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    build_dir = os.environ.get("CUDAQ_PULSE_BUILD_DIR", "")
    if build_dir:
        candidate = Path(
            build_dir) / "lib" / "Runtime" / "cudm" / "libcudm-runtime.so"
        if candidate.exists():
            return candidate

    search_paths = [
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]
    for p in search_paths:
        candidate = p / "libcudm-runtime.so"
        if candidate.exists():
            return candidate
    return None


def _find_mlir_opt() -> Optional[Path]:
    """Locate cudaq-pulse-opt (from build dir or PATH)."""
    build_dir = os.environ.get("CUDAQ_PULSE_BUILD_DIR", "")
    if build_dir:
        for subpath in ("core/mlir/tools/cudaq-pulse-opt/cudaq-pulse-opt",
                        "bin/cudaq-pulse-opt"):
            candidate = Path(build_dir) / subpath
            if candidate.exists():
                return candidate

    mlir_dir = os.environ.get("MLIR_DIR", "")
    if mlir_dir:
        candidate = Path(mlir_dir) / ".." / ".." / "bin" / "mlir-opt"
        if candidate.exists():
            return candidate.resolve()

    for p in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("cudaq-pulse-opt", "mlir-opt"):
            candidate = Path(p) / name
            if candidate.exists():
                return candidate
    return None


def _check_gpu_available() -> bool:
    """Check whether an NVIDIA GPU is accessible."""
    cuda_rt = _find_cuda_runtime()
    if cuda_rt is None:
        return False
    try:
        lib = ctypes.CDLL(str(cuda_rt))
        count = ctypes.c_int(0)
        err = lib.cudaGetDeviceCount(ctypes.byref(count))
        return err == 0 and count.value > 0
    except OSError:
        return False


@dataclass(frozen=True)
class JITResult:
    """Container for values returned from JIT-compiled code."""

    raw_ptr: int
    shape: tuple[int, ...]
    dtype: np.dtype

    def to_numpy(self) -> np.ndarray:
        """Interpret the raw device-to-host copy as a NumPy array."""
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        buf = (ctypes.c_double * (n_elements * 2)).from_address(self.raw_ptr)
        flat = np.frombuffer(buf, dtype=np.complex128, count=n_elements)
        return flat.reshape(self.shape).copy()


def _try_native_pipeline(pulse_mlir: str) -> Optional[str]:
    """Try to use the native nanobind pipeline if available."""
    try:
        from .._native._cudaq_pulse_native import MLIRPipeline
        pipeline = MLIRPipeline()
        return pipeline.run_full_pipeline(pulse_mlir)
    except ImportError:
        return None


def _run_mlir_opt_pipeline(pulse_mlir: str, work_dir: Path) -> str:
    """Run the MLIR pass pipeline via external mlir-opt process."""
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        raise FileNotFoundError("Cannot find mlir-opt or cudaq-pulse-opt. "
                                "Set CUDAQ_PULSE_BUILD_DIR or MLIR_DIR.")

    input_path = work_dir / "pulse_input.mlir"
    output_path = work_dir / "llvm_output.mlir"
    input_path.write_text(pulse_mlir)

    subprocess.run(
        [
            str(mlir_opt),
            "--pulse-to-qop",
            "--qop-to-cudm",
            "--cudm-to-llvm",
            str(input_path),
            "-o",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path.read_text()


class JITCompiler:
    """Compiles pulse MLIR through the full lowering pipeline and executes.

    Pipeline: pulse MLIR -> qop -> cudm -> LLVM IR -> .so -> execute
    """

    def __init__(self, *, mlir_bin_dir: Optional[str] = None):
        self._mlir_bin = Path(
            mlir_bin_dir) if mlir_bin_dir else self._find_mlir_bin()
        self._cudm_lib: Optional[ctypes.CDLL] = None

    @staticmethod
    def _find_mlir_bin() -> Path:
        mlir_dir = os.environ.get("MLIR_DIR", "")
        if mlir_dir:
            candidate = Path(mlir_dir) / ".." / ".." / "bin"
            if candidate.exists():
                return candidate.resolve()
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if (Path(p) / "mlir-translate").exists():
                return Path(p)
        raise FileNotFoundError(
            "Cannot find MLIR tools. Set MLIR_DIR or ensure mlir-translate is on PATH."
        )

    def _load_cudm_runtime(self) -> ctypes.CDLL:
        if self._cudm_lib is not None:
            return self._cudm_lib
        lib_path = _find_cudm_runtime()
        if lib_path is None:
            raise FileNotFoundError(
                "Cannot find libcudm-runtime.so. Set CUDM_RUNTIME_LIB.")
        self._cudm_lib = ctypes.CDLL(str(lib_path))
        return self._cudm_lib

    def _compile_to_so(self, llvm_ir: str, work_dir: Path) -> Path:
        """Lower LLVM IR text to a shared object."""
        ll_path = work_dir / "module.ll"
        obj_path = work_dir / "module.o"
        so_path = work_dir / "module.so"

        ll_path.write_text(llvm_ir)

        subprocess.run(
            [
                "llc", "-relocation-model=pic", "-filetype=obj",
                str(ll_path), "-o",
                str(obj_path)
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["clang", "-shared", "-o",
             str(so_path),
             str(obj_path)],
            check=True,
            capture_output=True,
        )
        return so_path

    def compile_pulse_mlir(self, pulse_mlir: str) -> Path:
        """Run full pipeline: pulse MLIR -> LLVM -> .so"""
        work_dir = Path(tempfile.mkdtemp(prefix="cudaq_pulse_jit_"))

        # Try native pipeline first, fall back to mlir-opt
        llvm_mlir = _try_native_pipeline(pulse_mlir)
        if llvm_mlir is None:
            llvm_mlir = _run_mlir_opt_pipeline(pulse_mlir, work_dir)

        # Translate LLVM MLIR to LLVM IR
        mlir_path = work_dir / "llvm_dialect.mlir"
        llvm_path = work_dir / "module.ll"
        mlir_path.write_text(llvm_mlir)

        translate = self._mlir_bin / "mlir-translate"
        subprocess.run(
            [
                str(translate), "--mlir-to-llvmir",
                str(mlir_path), "-o",
                str(llvm_path)
            ],
            check=True,
            capture_output=True,
        )
        llvm_ir = llvm_path.read_text()
        return self._compile_to_so(llvm_ir, work_dir)

    def compile_module(self, mlir_text: str) -> Path:
        """Translate MLIR (LLVM dialect) to a shared library (legacy API)."""
        work_dir = Path(tempfile.mkdtemp(prefix="cudaq_pulse_jit_"))
        mlir_path = work_dir / "module.mlir"
        llvm_path = work_dir / "module.ll"
        mlir_path.write_text(mlir_text)

        translate = self._mlir_bin / "mlir-translate"
        subprocess.run(
            [
                str(translate), "--mlir-to-llvmir",
                str(mlir_path), "-o",
                str(llvm_path)
            ],
            check=True,
            capture_output=True,
        )
        llvm_ir = llvm_path.read_text()
        return self._compile_to_so(llvm_ir, work_dir)

    def load_and_run(
        self,
        so_path: Path,
        entry: str = "main",
        args: Optional[Sequence[Any]] = None,
        n_qubits: int = 1,
    ) -> List[JITResult]:
        """Load a compiled .so and invoke its entry point."""
        self._load_cudm_runtime()
        module = ctypes.CDLL(str(so_path))
        func = getattr(module, entry)
        func.restype = ctypes.c_void_p
        c_args = _marshal_args(args or [])
        raw = func(*c_args)
        hilbert_dim = 2**n_qubits
        return [
            JITResult(raw_ptr=raw,
                      shape=(hilbert_dim,),
                      dtype=np.dtype(np.complex128))
        ]


def _marshal_args(args: Sequence[Any]) -> list:
    """Convert Python arguments to ctypes-compatible values."""
    c_args: list = []
    for a in args:
        if isinstance(a, (int, np.integer)):
            c_args.append(ctypes.c_int64(int(a)))
        elif isinstance(a, (float, np.floating)):
            c_args.append(ctypes.c_double(float(a)))
        elif isinstance(a, np.ndarray):
            c_args.append(a.ctypes.data_as(ctypes.c_void_p))
        else:
            c_args.append(ctypes.c_void_p(id(a)))
    return c_args


def compile_and_run(
    module_text: str,
    args: Optional[Sequence[Any]] = None,
    *,
    entry: str = "main",
    n_qubits: int = 1,
) -> List[JITResult]:
    """One-shot: compile an MLIR module and execute it.

    Raises ``RuntimeError`` if no GPU is available.
    """
    if not _check_gpu_available():
        raise RuntimeError(
            "No GPU available. cudaq-pulse requires an NVIDIA GPU.")
    compiler = JITCompiler()
    so_path = compiler.compile_module(module_text)
    return compiler.load_and_run(so_path,
                                 entry=entry,
                                 args=args or [],
                                 n_qubits=n_qubits)


def compile_and_run_pulse(
    pulse_mlir: str,
    *,
    entry: str = "main",
    n_qubits: int = 1,
) -> List[JITResult]:
    """One-shot: compile pulse MLIR through the full pipeline and execute.

    This is the new primary entry point that runs:
    pulse -> qop -> cudm -> llvm -> .so -> execute
    """
    if not _check_gpu_available():
        raise RuntimeError(
            "No GPU available. cudaq-pulse requires an NVIDIA GPU.")
    compiler = JITCompiler()
    so_path = compiler.compile_pulse_mlir(pulse_mlir)
    return compiler.load_and_run(so_path, entry=entry, n_qubits=n_qubits)
