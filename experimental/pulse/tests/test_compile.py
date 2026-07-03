# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cudaq_pulse.compile() public API."""

from __future__ import annotations

import math

import pytest

import cudaq_pulse as pulse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pulse.kernel
def _bell(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)
    sync(d0, d1)
    drive(d1, wf, t1)


_FREQ_2Q = {0: 5.0e9, 1: 5.1e9}

# ---------------------------------------------------------------------------
# Basic compile() usage
# ---------------------------------------------------------------------------


def test_compile_returns_compiled_kernel():
    ck = pulse.compile(
        _bell,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz=_FREQ_2Q,
    )
    assert isinstance(ck, pulse.CompiledKernel)


def test_compile_produces_mlir():
    ck = pulse.compile(
        _bell,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz=_FREQ_2Q,
    )
    mlir = ck.mlir
    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.gaussian" in mlir
    assert "pulse.drive" in mlir


def test_compile_metrics():
    ck = pulse.compile(
        _bell,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz=_FREQ_2Q,
    )
    m = ck.metrics
    assert isinstance(m, pulse.CompileMetrics)
    assert m.total_ms > 0
    assert m.trace_ms > 0
    assert m.ffi_ms > 0


def test_compile_no_passes():
    ck = pulse.compile(
        _bell,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz=_FREQ_2Q,
        passes=(),
    )
    assert ck.mlir is not None
    assert "pulse.drive" in ck.mlir


def test_compile_single_qubit():

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf, t)

    ck = pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5.0e9})
    assert "pulse.drive" in ck.mlir


def test_compile_with_virtual_z():

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        shift_phase(t, math.pi / 4)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf, t)

    ck = pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5.0e9})
    assert ck.mlir is not None
    assert ck.metrics.total_ms > 0


def test_compile_with_fusion():

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        sq1 = square(50, 0.2)
        drive(d, sq1, t)
        sq2 = square(50, 0.2)
        drive(d, sq2, t)

    ck = pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5.0e9})
    assert ck.mlir is not None


def test_compile_bad_schedule_raises():
    with pytest.raises(ValueError, match="Unknown schedule"):
        pulse.compile(_bell,
                      [pulse.qudit_ref(), pulse.qudit_ref()],
                      qubit_freq_hz=_FREQ_2Q,
                      schedule="bogus")


def test_compile_no_args_raises():
    with pytest.raises(TypeError, match="requires args"):
        pulse.compile(_bell, qubit_freq_hz=_FREQ_2Q)


# ---------------------------------------------------------------------------
# PackedIRBuilder tests
# ---------------------------------------------------------------------------


def test_packed_ir_builder_basic():
    """PackedIRBuilder produces valid int64 buffer."""
    import numpy as np
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    (q,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (dl, t) = b.emit("pulse.get_drive_line", (q,), ("drive_line", "tone"))
    (wf,) = b.emit("pulse.gaussian", (), ("waveform",), {
        "duration": 40,
        "amplitude": 0.5,
        "sigma": 10.0
    })
    b.emit("pulse.drive", (dl, wf, t), ("drive_line", "tone"))

    buf = b.get_buffer()
    assert isinstance(buf, np.ndarray)
    assert buf.dtype == np.int64
    assert len(buf) > 0
    assert b.n_qubits == 1


def test_packed_ir_builder_readout():
    """PackedIRBuilder encodes readout ops."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    b = PackedIRBuilder(clock_ghz=2.0, qubit_freq_hz={0: 5e9})
    (q,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (rl, t) = b.emit("pulse.get_readout_line", (q,), ("readout_line", "tone"))
    (wf,) = b.emit("pulse.square", (), ("waveform",), {
        "duration": 400,
        "amplitude": 0.1
    })
    b.emit("pulse.readout", (rl, wf, t),
           ("readout_line", "tone", "measurement"))

    buf = b.get_buffer()
    assert (buf[0] & 0xFF) == 1  # ALLOC_READOUT
    assert len(buf) > 8


def test_packed_ir_builder_sync():
    """PackedIRBuilder encodes variable-length sync ops."""
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder
    from cudaq_pulse.kernel.ir_builder import IRValue

    b = PackedIRBuilder(clock_ghz=2.0, qubit_freq_hz={0: 5e9, 1: 5.1e9})
    (q0,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (q1,) = b.emit("pulse.qudit_alloc", (), ("qref",))
    (d0, t0) = b.emit("pulse.get_drive_line", (q0,), ("drive_line", "tone"))
    (d1, t1) = b.emit("pulse.get_drive_line", (q1,), ("drive_line", "tone"))
    b.emit("pulse.sync", (d0, d1), ("drive_line", "drive_line"))

    buf = b.get_buffer()
    assert len(buf) > 0


def test_compile_module_available():
    """compile() produces an in-memory PulseModule."""
    ck = pulse.compile(
        _bell,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz=_FREQ_2Q,
    )
    assert ck.module is not None
    assert ck.mlir is not None
