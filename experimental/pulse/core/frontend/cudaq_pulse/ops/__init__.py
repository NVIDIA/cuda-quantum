# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel DSL operations for pulse programming.

These functions form the "language extension" vocabulary used as bare
names inside ``@pulse.kernel`` decorated functions.  During bytecode
tracing the decorator intercepts calls to these names and lowers them
to MLIR ops.  If called outside a kernel context they raise
``RuntimeError``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Union

from ..kernel.ir_builder import Parameter

Numeric = Union[int, float, Parameter]

from ._context import get_active_context

# ── Opaque type aliases (for IDE hover-docs, not enforced at runtime) ────


class Waveform:
    """Opaque waveform handle returned by envelope constructors."""


class Line:
    """Opaque drive or readout line handle from ``get_drive_line()``."""


class Tone:
    """Opaque tone handle for phase and frequency operations."""


class MeasurementResult:
    """Opaque measurement result from ``readout()``."""


def _require_context(op_name: str) -> Any:
    ctx = get_active_context()
    if ctx is None:
        raise RuntimeError(
            f"{op_name}() must be called inside a @cudaq_pulse.kernel function")
    return ctx


# ── Channel access ───────────────────────────────────────────────────────


def get_drive_line(qubit: Any) -> tuple[Line, Tone]:
    """Obtain the drive line and tone for a qubit.

    Args:
        qubit: A ``QuditRef`` identifying the target qubit.

    Returns:
        A ``(line, tone)`` tuple used by ``drive()`` and phase/frequency ops.
    """
    return _require_context("get_drive_line")


def get_readout_line(qubit: Any) -> tuple[Line, Tone]:
    """Obtain the readout line and tone for a qubit.

    Args:
        qubit: A ``QuditRef`` identifying the target qubit.

    Returns:
        A ``(line, tone)`` tuple used by ``readout()``.
    """
    return _require_context("get_readout_line")


# ── Scheduling ops ──────────────────────────────────────────────────────


def drive(line: Line, waveform: Waveform, tone: Tone) -> None:
    """Play a waveform on a drive line.

    Args:
        line: Drive line from ``get_drive_line()``.
        waveform: Waveform envelope to play (e.g. from ``gaussian()``).
        tone: Tone handle from ``get_drive_line()``.
    """
    _require_context("drive")


def readout(line: Line, waveform: Waveform, tone: Tone) -> MeasurementResult:
    """Acquire a measurement through a readout line.

    Args:
        line: Readout line from ``get_readout_line()``.
        waveform: Readout waveform envelope.
        tone: Tone handle from ``get_readout_line()``.

    Returns:
        Measurement result handle.
    """
    _require_context("readout")


def wait(target: Line, duration: Numeric) -> None:
    """Insert an idle delay on a line.

    Args:
        target: Drive or readout line.
        duration: Wait duration in clock cycles.
    """
    _require_context("wait")


def sync(*targets: Line) -> None:
    """Synchronize multiple lines to a common time point.

    All lines are padded to the latest time among them before
    subsequent operations proceed.

    Args:
        targets: Two or more drive/readout lines to synchronize.
    """
    _require_context("sync")


# ── Phase / frequency ops ───────────────────────────────────────────────


def shift_phase(tone: Tone, phase: Numeric) -> None:
    """Add a relative phase offset to a tone's rotating frame.

    Args:
        tone: Tone handle (second element of ``get_drive_line()``).
        phase: Phase increment in radians.
    """
    _require_context("shift_phase")


def set_phase(tone: Tone, phase: Numeric) -> None:
    """Set the absolute phase of a tone's rotating frame.

    Args:
        tone: Tone handle.
        phase: Absolute phase in radians.
    """
    _require_context("set_phase")


def shift_frequency(tone: Tone, frequency: Numeric) -> None:
    """Add a relative frequency offset to a tone.

    Args:
        tone: Tone handle.
        frequency: Frequency offset in Hz.
    """
    _require_context("shift_frequency")


def set_frequency(tone: Tone, frequency: Numeric) -> None:
    """Set the absolute frequency of a tone.

    Args:
        tone: Tone handle.
        frequency: Absolute frequency in Hz.
    """
    _require_context("set_frequency")


# ── Waveform constructors ───────────────────────────────────────────────


def gaussian(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a Gaussian envelope waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Standard deviation in clock cycles.

    Returns:
        Waveform value for use with ``drive()`` or waveform arithmetic.
    """
    return _require_context("gaussian")


def square(duration: Numeric, amplitude: Numeric) -> Waveform:
    """Create a flat-top (square) envelope waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Constant amplitude in ``[-1, 1]``.

    Returns:
        Waveform value.
    """
    return _require_context("square")


def drag(
    duration: Numeric,
    amplitude: Numeric,
    sigma: Numeric,
    beta: Numeric,
) -> Waveform:
    """Create a DRAG (Derivative Removal by Adiabatic Gate) waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Gaussian standard deviation in clock cycles.
        beta: DRAG correction coefficient.

    Returns:
        Waveform value.
    """
    return _require_context("drag")


def cosine(duration: Numeric, amplitude: Numeric) -> Waveform:
    """Create a raised-cosine envelope waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.

    Returns:
        Waveform value.
    """
    return _require_context("cosine")


def tanh_ramp(duration: Numeric, amplitude: Numeric, sigma: Numeric) -> Waveform:
    """Create a hyperbolic-tangent ramp waveform.

    Args:
        duration: Pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Rise/fall steepness in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("tanh_ramp")


def gaussian_square(
    duration: Numeric,
    amplitude: Numeric,
    sigma: Numeric,
    width: Numeric,
) -> Waveform:
    """Create a Gaussian-square (flat-top Gaussian) waveform.

    A square pulse with Gaussian rise and fall edges.

    Args:
        duration: Total pulse duration in clock cycles.
        amplitude: Peak amplitude in ``[-1, 1]``.
        sigma: Gaussian edge standard deviation in clock cycles.
        width: Flat-top width in clock cycles.

    Returns:
        Waveform value.
    """
    return _require_context("gaussian_square")


def custom(duration: int, envelope_fn: Callable[..., complex]) -> Waveform:
    """Create a waveform from a callable envelope function.

    Args:
        duration: Pulse duration in clock cycles.
        envelope_fn: Callable ``f(t) -> complex`` defining the envelope.

    Returns:
        Waveform value.
    """
    return _require_context("custom")


def custom_samples(samples: Sequence[complex]) -> Waveform:
    """Create a waveform from pre-computed sample data.

    Args:
        samples: Array-like of complex sample values.

    Returns:
        Waveform value.
    """
    return _require_context("custom_samples")


# ── Waveform arithmetic ─────────────────────────────────────────────────


def wf_add(left: Waveform, right: Waveform) -> Waveform:
    """Add two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform (must have same duration as *left*).

    Returns:
        Combined waveform ``left + right``.
    """
    return _require_context("wf_add")


def wf_sub(left: Waveform, right: Waveform) -> Waveform:
    """Subtract two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform.

    Returns:
        Difference waveform ``left - right``.
    """
    return _require_context("wf_sub")


def wf_mul(left: Waveform, right: Waveform) -> Waveform:
    """Multiply two waveforms element-wise.

    Args:
        left: First waveform.
        right: Second waveform.

    Returns:
        Product waveform ``left * right``.
    """
    return _require_context("wf_mul")


def wf_scale(scalar: float, waveform: Waveform) -> Waveform:
    """Scale a waveform by a constant factor.

    Args:
        scalar: Real scaling factor.
        waveform: Waveform to scale.

    Returns:
        Scaled waveform ``scalar * waveform``.
    """
    return _require_context("wf_scale")


def wf_neg(waveform: Waveform) -> Waveform:
    """Negate a waveform (flip sign of all samples).

    Args:
        waveform: Waveform to negate.

    Returns:
        Negated waveform ``-waveform``.
    """
    return _require_context("wf_neg")
