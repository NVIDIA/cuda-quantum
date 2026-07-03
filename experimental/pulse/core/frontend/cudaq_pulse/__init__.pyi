# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type stub for cudaq_pulse -- provides IDE autocomplete and hover docs."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

Numeric = Union[int, float, "Parameter"]

__version__: str

# ── Opaque DSL types ─────────────────────────────────────────────────────


class Waveform:
    """Opaque waveform handle returned by envelope constructors."""
    ...


class Line:
    """Opaque drive or readout line handle."""
    ...


class Tone:
    """Opaque tone handle for phase and frequency operations."""
    ...


class MeasurementResult:
    """Opaque measurement result from ``readout()``."""
    ...


class Parameter:
    """Sentinel for a symbolic kernel parameter (compile-once, evaluate-many).

    Parameters are automatically created by ``compile()`` for non-qudit
    kernel arguments. They can be passed directly to pulse ops like
    ``gaussian(64, amplitude, 16.0)`` where ``amplitude`` is a Parameter.
    """
    name: str
    index: int
    dtype: str

    def __init__(self, name: str, index: int, dtype: str = "f64") -> None:
        ...


# ── Infrastructure ───────────────────────────────────────────────────────


class QuditRef:
    """A single qudit reference representing one quantum degree of freedom."""

    def __init__(self) -> None:
        ...


class QvecRef:
    """A fixed-size vector of qudit references."""

    def __init__(self, size: int) -> None:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> QuditRef:
        ...


def qudit_ref() -> QuditRef:
    """Create a single qudit reference for use as a kernel argument."""
    ...


def qvec_ref(size: int) -> QvecRef:
    """Create a vector of *size* qudit references."""
    ...


def kernel(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that marks a Python function as a pulse kernel.

    The decorated function is traced via bytecode capture when called,
    producing an intermediate representation that can be compiled to
    MLIR with ``pulse.compile()``.
    """
    ...


class CompileMetrics:
    """Per-stage timing breakdown (all values in milliseconds)."""
    trace_ms: float
    ffi_ms: float
    passes_ms: float
    schedule_ms: float
    total_ms: float
    op_count: int


class CompiledKernel:
    """Result of ``pulse.compile()``.

    For parametric kernels, call the instance to evaluate at concrete
    values: ``compiled(amplitude=0.5)`` or ``compiled(0.5)``.
    """
    metrics: CompileMetrics

    @property
    def mlir(self) -> str:
        """Return the MLIR text representation of the compiled module."""
        ...

    @property
    def parameters(self) -> list[str]:
        """Names of symbolic parameters (empty for concrete kernels)."""
        ...

    @property
    def is_parametric(self) -> bool:
        """True if this kernel has symbolic parameters."""
        ...

    def __call__(self, *args: float | int, **kwargs: float | int) -> "CompiledKernel":
        """Evaluate a parametric kernel at concrete values.

        Returns a new fully-scheduled CompiledKernel.
        """
        ...


def compile(
    kernel_fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    qubit_freq_hz: dict[int, float] | None = None,
    passes: Sequence[str] | None = None,
    schedule: str = "alap",
) -> CompiledKernel:
    """Compile a ``@pulse.kernel`` function into a scheduled MLIR module.

    Args:
        kernel_fn: A ``@pulse.kernel``-decorated function.
        args: Positional arguments (typically ``pulse.qudit_ref()`` objects).
        qubit_freq_hz: Mapping of qubit index to frequency in Hz.
        passes: Optimization passes to run (default: verify, virtual_z, fusion).
        schedule: Scheduling strategy (``"alap"``, ``"asap"``, ``"rcp"``).

    Returns:
        A ``CompiledKernel`` with MLIR text and compile metrics.
    """
    ...


# ── Kernel DSL: channel access ───────────────────────────────────────────


def get_drive_line(qubit: QuditRef) -> tuple[Line, Tone]:
    """Obtain the drive line and tone for a qubit."""
    ...


def get_readout_line(qubit: QuditRef) -> tuple[Line, Tone]:
    """Obtain the readout line and tone for a qubit."""
    ...


# ── Kernel DSL: scheduling ops ──────────────────────────────────────────


def drive(line: Line, waveform: Waveform, tone: Tone) -> None:
    """Play a waveform on a drive line."""
    ...


def readout(line: Line, waveform: Waveform, tone: Tone) -> MeasurementResult:
    """Acquire a measurement through a readout line."""
    ...


def wait(target: Line, duration: Numeric) -> None:
    """Insert an idle delay on a line."""
    ...


def sync(*targets: Line) -> None:
    """Synchronize multiple lines to a common time point."""
    ...


# ── Kernel DSL: phase / frequency ───────────────────────────────────────


def shift_phase(tone: Tone, phase: Numeric) -> None:
    """Add a relative phase offset to a tone's rotating frame."""
    ...


def set_phase(tone: Tone, phase: Numeric) -> None:
    """Set the absolute phase of a tone's rotating frame."""
    ...


def shift_frequency(tone: Tone, frequency: Numeric) -> None:
    """Add a relative frequency offset to a tone."""
    ...


def set_frequency(tone: Tone, frequency: Numeric) -> None:
    """Set the absolute frequency of a tone."""
    ...


# ── Kernel DSL: waveform constructors ───────────────────────────────────


def gaussian(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a Gaussian envelope waveform."""
    ...


def square(duration: Numeric, amplitude: Numeric) -> Waveform:
    """Create a flat-top (square) envelope waveform."""
    ...


def drag(duration: Numeric, amplitude: Numeric, sigma: Numeric,
         beta: Numeric) -> Waveform:
    """Create a DRAG waveform."""
    ...


def cosine(duration: Numeric, amplitude: Numeric) -> Waveform:
    """Create a raised-cosine envelope waveform."""
    ...


def tanh_ramp(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a hyperbolic-tangent ramp waveform."""
    ...


def gaussian_square(duration: Numeric, amplitude: Numeric, sigma: Numeric,
                    width: Numeric) -> Waveform:
    """Create a Gaussian-square (flat-top Gaussian) waveform."""
    ...


def custom(duration: int, envelope_fn: Callable[..., complex]) -> Waveform:
    """Create a waveform from a callable envelope function."""
    ...


def custom_samples(samples: Sequence[complex]) -> Waveform:
    """Create a waveform from pre-computed sample data."""
    ...


# ── Kernel DSL: waveform arithmetic ─────────────────────────────────────


def wf_add(left: Waveform, right: Waveform) -> Waveform:
    """Add two waveforms element-wise."""
    ...


def wf_sub(left: Waveform, right: Waveform) -> Waveform:
    """Subtract two waveforms element-wise."""
    ...


def wf_mul(left: Waveform, right: Waveform) -> Waveform:
    """Multiply two waveforms element-wise."""
    ...


def wf_scale(scalar: float, waveform: Waveform) -> Waveform:
    """Scale a waveform by a constant factor."""
    ...


def wf_neg(waveform: Waveform) -> Waveform:
    """Negate a waveform."""
    ...
