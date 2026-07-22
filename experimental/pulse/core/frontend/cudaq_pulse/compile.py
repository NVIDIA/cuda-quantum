# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public compile entry point for cudaq-pulse.

``cudaq_pulse.compile()`` is the single public way to turn a
``@cudaq_pulse.kernel`` into a scheduled, MLIR-backed compilation
artifact.  Native bindings are **required** -- there is no fallback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

try:
    from _cudaq_pulse_native import PulseModuleBuilder
except ImportError as _e:
    raise RuntimeError(
        "cudaq-pulse native bindings required. "
        "Build with: cd cudaq-pulse && mkdir build && cd build && "
        "cmake .. -G Ninja && ninja") from _e

from .kernel.ir_builder import Parameter
from .passes.scheduling import ScheduleMetrics, MachineModel

_VALID_SCHEDULES = frozenset({"asap", "alap", "rcp", "alap_rcp"})

# C++ pass names for the full pipeline
_PASS_MAP = {
    "verify": "pulse-verify",
    "canonicalize": "pulse-canonicalize",
    "virtual_z": "pulse-virtual-z",
    "fusion": "pulse-fusion",
    "licm": "loop-invariant-code-motion",
}

_SCHEDULE_MAP = {
    "alap": "pulse-schedule-alap",
}

DEFAULT_PASSES: Tuple[str, ...] = (
    "verify",
    "virtual_z",
    "fusion",
)

# ---------------------------------------------------------------------------
# CompiledKernel
# ---------------------------------------------------------------------------


@dataclass()
class CompileMetrics:
    """Per-stage timing breakdown (all values in milliseconds)."""
    trace_ms: float = 0.0
    ffi_ms: float = 0.0
    passes_ms: float = 0.0
    schedule_ms: float = 0.0
    total_ms: float = 0.0
    op_count: int = 0
    # Legacy fields for backward compat with benchmarks
    capture_ms: float = 0.0
    lower_ms: float = 0.0
    mlir_emit_ms: float = 0.0
    schedule_metrics: Optional[ScheduleMetrics] = None


@dataclass()
class CompiledKernel:
    """Result of ``cudaq_pulse.compile()``.

    Holds an in-memory ``PulseModule`` (MLIR ModuleOp) and provides
    access to MLIR text, lowering, and GPU execution.

    For parametric kernels, call the instance to evaluate at concrete
    values: ``compiled(amplitude=0.5)`` or ``compiled(0.5)``.
    """
    _pulse_module: Any = field(default=None, repr=False)
    _mlir_text: Optional[str] = field(default=None, repr=False)
    metrics: CompileMetrics = field(default_factory=CompileMetrics)
    _param_names: list = field(default_factory=list, repr=False)
    _param_types: list = field(default_factory=list, repr=False)

    @property
    def mlir(self) -> str:
        """Pulse-dialect MLIR text."""
        if self._mlir_text is None and self._pulse_module is not None:
            self._mlir_text = self._pulse_module.print()
        return self._mlir_text or ""

    @property
    def module(self):
        """The in-memory PulseModule."""
        return self._pulse_module

    @property
    def parameters(self) -> list[str]:
        """Names of symbolic parameters (empty for concrete kernels)."""
        return list(self._param_names)

    @property
    def is_parametric(self) -> bool:
        return bool(self._param_names)

    def __call__(self, *args, **kwargs) -> "CompiledKernel":
        """Evaluate a parametric kernel at concrete values.

        Accepts positional or keyword arguments matching the parameter names.
        Returns a new fully-scheduled ``CompiledKernel`` with concrete MLIR.

        Raises ``TypeError`` for non-parametric kernels or argument mismatches.
        """
        if not self._param_names:
            raise TypeError(
                "This kernel has no parameters. "
                "Only parametric kernels can be evaluated with (...).")

        values = self._resolve_args(args, kwargs)

        f64_vals: list[float] = []
        i64_vals: list[int] = []
        for val, dtype in zip(values, self._param_types):
            if dtype == "i64":
                i64_vals.append(int(val))
            else:
                f64_vals.append(float(val))

        t0 = time.perf_counter()
        new_module = self._pulse_module.specialize(f64_vals, i64_vals)
        specialize_ms = (time.perf_counter() - t0) * 1000

        return CompiledKernel(
            _pulse_module=new_module,
            metrics=CompileMetrics(schedule_ms=specialize_ms,
                                   total_ms=specialize_ms),
        )

    def _resolve_args(self, args: tuple, kwargs: dict) -> list:
        """Merge positional and keyword args into an ordered value list."""
        if args and kwargs:
            raise TypeError(
                "Cannot mix positional and keyword arguments. "
                "Use either compiled(0.5, 64) or compiled(amplitude=0.5, duration=64)."
            )
        if kwargs:
            values = []
            for name in self._param_names:
                if name not in kwargs:
                    raise TypeError(
                        f"Missing parameter {name!r}. "
                        f"Required: {', '.join(self._param_names)}")
                values.append(kwargs[name])
            extra = set(kwargs) - set(self._param_names)
            if extra:
                raise TypeError(
                    f"Unknown parameters: {', '.join(sorted(extra))}. "
                    f"Available: {', '.join(self._param_names)}")
            return values
        if len(args) != len(self._param_names):
            raise TypeError(
                f"Expected {len(self._param_names)} arguments "
                f"({', '.join(self._param_names)}), got {len(args)}")
        return list(args)

    def lower_to_llvm(self) -> str:
        """Run full MLIR lowering (pulse -> qop -> cudm -> llvm)."""
        if self._pulse_module is not None:
            return self._pulse_module.run_full_lowering()
        raise RuntimeError("No PulseModule available")

    def run(self, *, entry: str = "main", n_qubits: Optional[int] = None):
        """JIT-compile and execute on GPU via cuDensityMat."""
        from .runtime.jit import compile_and_run_pulse
        return compile_and_run_pulse(self.mlir,
                                     entry=entry,
                                     n_qubits=n_qubits or 1)


# ---------------------------------------------------------------------------
# compile()
# ---------------------------------------------------------------------------


def compile(
    kernel_fn,
    args: Sequence[Any] | None = None,
    *,
    clock_ghz: float = 2.0,
    qubit_freq_hz: dict[int, float] | None = None,
    schedule: str = "alap",
    passes: Sequence[str] | None = None,
    machine: MachineModel | None = None,
) -> CompiledKernel:
    """Compile a ``@cudaq_pulse.kernel`` into a scheduled MLIR module.

    This is the only public compilation entry point.  It traces the kernel
    directly into a packed int64 buffer, sends it to C++ in a single
    zero-copy FFI call, and runs all passes on the in-memory MLIR module.

    Parameters
    ----------
    kernel_fn:
        A ``@cudaq_pulse.kernel``-decorated function.
    args:
        Arguments (typically ``qudit_ref()`` objects).
    clock_ghz:
        System clock frequency in GHz.
    qubit_freq_hz:
        Mapping from qubit index to frequency in Hz.
    schedule:
        Scheduling policy: ``"asap"``, ``"alap"``, ``"rcp"``, ``"alap_rcp"``.
    passes:
        Optimization passes to run.  Default is ``DEFAULT_PASSES``.
        Pass ``()`` to skip.
    machine:
        Machine model for resource-constrained scheduling.
    """
    if schedule not in _VALID_SCHEDULES:
        raise ValueError(f"Unknown schedule policy {schedule!r}. "
                         f"Choose from: {', '.join(sorted(_VALID_SCHEDULES))}")

    if passes is None:
        passes = DEFAULT_PASSES

    freq = qubit_freq_hz or {}
    metrics = CompileMetrics()

    if not callable(kernel_fn):
        raise TypeError(f"Expected a @cudaq_pulse.kernel function; "
                        f"got {type(kernel_fn).__name__}")

    if args is None:
        raise TypeError(
            "compile() requires args= when passing a kernel function. "
            "e.g. compile(my_kernel, [qudit_ref(), qudit_ref()], ...)")

    # Detect parameter args: kernel args beyond the qudit args are parameters
    fn = getattr(kernel_fn, "__wrapped__", kernel_fn)
    code = fn.__code__
    all_params = list(code.co_varnames[:code.co_argcount])
    from .kernel.decorator import QuditRef, QvecRef
    qudit_args = [a for a in args if isinstance(a, (QuditRef, QvecRef))]
    param_names = all_params[len(qudit_args):]

    if param_names:
        return _compile_parametric(fn, qudit_args, param_names, clock_ghz,
                                   freq, passes, schedule, metrics)

    # Step 1: Trace kernel directly into packed buffer
    t0 = time.perf_counter()
    buf, n_qubits, freq_arr = _trace_to_packed(kernel_fn, args, clock_ghz, freq)
    metrics.trace_ms = (time.perf_counter() - t0) * 1000
    metrics.capture_ms = metrics.trace_ms

    # Step 2: Single FFI call -- build in-memory MLIR module
    t0 = time.perf_counter()
    builder = PulseModuleBuilder()
    pulse_module = builder.build_from_packed(buf, clock_ghz, n_qubits, freq_arr)
    metrics.ffi_ms = (time.perf_counter() - t0) * 1000
    metrics.mlir_emit_ms = metrics.ffi_ms

    # Step 3: Run C++ passes on the in-memory module
    t0 = time.perf_counter()
    cpp_passes = [_PASS_MAP[p] for p in passes if p in _PASS_MAP]
    if cpp_passes:
        pulse_module.run_passes(cpp_passes)
    metrics.passes_ms = (time.perf_counter() - t0) * 1000

    # Step 4: Schedule via C++ pass
    t0 = time.perf_counter()
    sched_pass = _SCHEDULE_MAP.get(schedule)
    if sched_pass:
        pulse_module.run_passes([sched_pass])
    metrics.schedule_ms = (time.perf_counter() - t0) * 1000

    metrics.total_ms = (metrics.trace_ms + metrics.ffi_ms + metrics.passes_ms +
                        metrics.schedule_ms)

    return CompiledKernel(
        _pulse_module=pulse_module,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _compile_parametric(
    fn,
    qudit_args: list,
    param_names: list[str],
    clock_ghz: float,
    freq: dict[int, float],
    passes: Sequence[str],
    schedule: str,
    metrics: CompileMetrics,
) -> CompiledKernel:
    """Compile a parametric kernel (compile-once, evaluate-many)."""
    from .kernel.packed_ir_builder import PackedIRBuilder
    from .kernel.bytecode_bridge import _trace_kernel_with_builder

    # Create Parameter sentinels for each non-qudit argument
    param_sentinels = []
    param_types = []
    for i, name in enumerate(param_names):
        dtype = "f64"  # default to f64; i64 if name suggests integer
        if "duration" in name or "dur" in name or "delay" in name:
            dtype = "i64"
        param_sentinels.append(Parameter(name, i, dtype))
        param_types.append(dtype)

    # Trace with Parameter sentinels in place of concrete values
    t0 = time.perf_counter()
    name = getattr(fn, "__name__", "kernel")
    builder = PackedIRBuilder(name=name,
                              clock_ghz=clock_ghz,
                              qubit_freq_hz=freq)

    trace_args = list(qudit_args) + param_sentinels
    _trace_kernel_with_builder(fn, builder, trace_args)

    buf = builder.get_buffer()
    n_qubits = builder.n_qubits
    freq_arr = builder.get_freq_array()
    metrics.trace_ms = (time.perf_counter() - t0) * 1000
    metrics.capture_ms = metrics.trace_ms

    # Build parametric MLIR module (func.func with block args)
    t0 = time.perf_counter()
    mlir_builder = PulseModuleBuilder()
    pulse_module = mlir_builder.build_from_packed(buf, clock_ghz, n_qubits,
                                                  freq_arr, param_names,
                                                  param_types)
    metrics.ffi_ms = (time.perf_counter() - t0) * 1000
    metrics.mlir_emit_ms = metrics.ffi_ms

    # Run structural passes (no scheduling -- deferred to evaluate)
    t0 = time.perf_counter()
    cpp_passes = [_PASS_MAP[p] for p in passes if p in _PASS_MAP]
    if cpp_passes:
        pulse_module.run_passes(cpp_passes)
    metrics.passes_ms = (time.perf_counter() - t0) * 1000

    # Do NOT schedule parametric kernels -- scheduling needs concrete durations
    metrics.total_ms = metrics.trace_ms + metrics.ffi_ms + metrics.passes_ms

    return CompiledKernel(
        _pulse_module=pulse_module,
        metrics=metrics,
        _param_names=param_names,
        _param_types=param_types,
    )


def _trace_to_packed(kernel_fn, args, clock_ghz: float,
                     freq: dict[int, float]) -> tuple[Any, int, Any]:
    """Trace the kernel into a packed int64 numpy buffer.

    Returns (buffer, n_qubits, freq_array).
    """
    from .kernel.packed_ir_builder import PackedIRBuilder
    from .kernel.bytecode_bridge import _trace_kernel_with_builder

    if not callable(kernel_fn):
        raise TypeError(f"Expected a @cudaq_pulse.kernel function; "
                        f"got {type(kernel_fn).__name__}")

    if args is None:
        raise TypeError(
            "compile() requires args= when passing a kernel function. "
            "e.g. compile(my_kernel, [qudit_ref(), qudit_ref()], ...)")

    fn = getattr(kernel_fn, "__wrapped__", kernel_fn)
    name = getattr(fn, "__name__", "kernel")
    builder = PackedIRBuilder(name=name,
                              clock_ghz=clock_ghz,
                              qubit_freq_hz=freq)

    _trace_kernel_with_builder(fn, builder, args)

    return builder.get_buffer(), builder.n_qubits, builder.get_freq_array()
