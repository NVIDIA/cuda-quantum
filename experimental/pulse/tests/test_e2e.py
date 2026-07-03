# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end roundtrip tests: kernel -> compile() -> CompiledKernel."""

from __future__ import annotations

import math

import cudaq_pulse as pulse


def test_single_qubit_full_pipeline():
    """Single qubit: kernel -> compile -> verify scheduled MLIR output."""

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        shift_phase(t, math.pi / 4)
        wf = gaussian(40, 0.3, 10.0)
        drive(d, wf, t)
        shift_phase(t, math.pi / 4)
        wf2 = gaussian(40, 0.3, 10.0)
        drive(d, wf2, t)

    ck = pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5.0e9})
    assert isinstance(ck, pulse.CompiledKernel)
    assert "pulse.drive" in ck.mlir
    assert ck.metrics.total_ms > 0
    assert ck.metrics.trace_ms > 0


def test_two_qubit_full_pipeline():
    """Two qubits with sync: full compilation pipeline via compile()."""

    @pulse.kernel
    def k(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)
        sync(d0, d1)
        drive(d1, wf, t1)

    ck = pulse.compile(
        k,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz={
            0: 5e9,
            1: 5.1e9
        },
    )
    assert isinstance(ck, pulse.CompiledKernel)
    assert "pulse.drive" in ck.mlir
    assert ck.metrics.total_ms > 0


def test_loop_full_pipeline():
    """Loop kernel: compile() handles loops end-to-end."""

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        for _ in range(5):
            wf = gaussian(40, 0.3, 10.0)
            drive(d, wf, t)
            wait(d, 20)

    ck = pulse.compile(k, [pulse.qudit_ref()], qubit_freq_hz={0: 5e9})
    assert isinstance(ck, pulse.CompiledKernel)
    assert "pulse.drive" in ck.mlir
    assert ck.metrics.total_ms > 0


def test_scheduling_pipeline():
    """Scheduling via compile() with ALAP policy.

    Note: RCP scheduling is not yet available in the C++ pass pipeline
    (_SCHEDULE_MAP only maps "alap"). This test uses schedule="alap".
    """

    @pulse.kernel
    def k(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)
        drive(d1, wf, t1)

    ck = pulse.compile(
        k,
        [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz={
            0: 5e9,
            1: 5.1e9
        },
        schedule="alap",
    )
    assert isinstance(ck, pulse.CompiledKernel)
    assert "pulse.drive" in ck.mlir
    assert ck.metrics.total_ms > 0
    assert ck.metrics.schedule_ms >= 0
