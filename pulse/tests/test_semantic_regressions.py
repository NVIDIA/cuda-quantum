# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Regression tests for semantic correctness at public API boundaries."""

from __future__ import annotations

import re

import numpy as np
import pytest

import cudaq_pulse as pulse
from cudaq_pulse._native._cudaq_pulse_native import PulseModuleBuilder
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.passes._builder import ProgramBuilder
from cudaq_pulse.passes.scheduling import (MachineModel, schedule_alap,
                                           schedule_asap, schedule_rcp)


def _drive_timings(mlir: str) -> list[tuple[int, int]]:
    timings = []
    for match in re.finditer(r"pulse\.drive.*?\{([^}]*)\}", mlir):
        attrs = match.group(1)
        start = re.search(r"start_vtu\s*=\s*(-?\d+)", attrs)
        duration = re.search(r"duration_vtu\s*=\s*(-?\d+)", attrs)
        assert start and duration
        timings.append((int(start.group(1)), int(duration.group(1))))
    return timings


def test_constant_loop_is_fully_unrolled():

    @pulse.kernel
    def kernel(q):
        line, tone = get_drive_line(q)
        waveform = square(10, 0.2)
        for _ in range(5):
            drive(line, waveform, tone)

    compiled = pulse.compile(kernel, [pulse.qudit_ref()],
                             qubit_freq_hz={0: 5.0e9},
                             schedule="asap",
                             passes=())
    assert compiled.mlir.count("= pulse.drive ") == 5
    assert _drive_timings(compiled.mlir) == [(0, 10), (10, 10), (20, 10),
                                             (30, 10), (40, 10)]


def test_range_start_stop_step_values_are_preserved():

    @pulse.kernel
    def kernel(q):
        _, tone = get_drive_line(q)
        for index in range(2, 9, 3):
            shift_phase(tone, float(index))

    compiled = pulse.compile(kernel, [pulse.qudit_ref()],
                             qubit_freq_hz={0: 5.0e9},
                             passes=())
    assert compiled.mlir.count("pulse.shift_phase") == 3
    for value in (2.0, 5.0, 8.0):
        assert f"{value:.6e}" in compiled.mlir


def test_measurement_dependent_branch_is_rejected():

    @pulse.kernel
    def kernel(q):
        line, tone = get_readout_line(q)
        result = readout(line, square(10, 0.2), tone)
        if result:
            wait(line, 10)

    with pytest.raises(CompilationError, match="runtime-dependent branches"):
        pulse.compile(kernel, [pulse.qudit_ref()], qubit_freq_hz={0: 5.0e9})


def test_documented_waveforms_and_algebra_lower_to_typed_ops():

    @pulse.kernel
    def kernel():
        q = pulse.qudit_ref()
        line, tone = get_drive_line(q)
        left = cosine(4, 0.2)
        right = custom_samples([0.1, 0.2, 0.3, 0.4])
        combined = wf_add(left, right)
        scaled = wf_scale(combined, 2.0)
        negated = wf_neg(scaled)
        drive(line, negated, tone)

    compiled = pulse.compile(kernel, [],
                             qubit_freq_hz={0: 5.0e9},
                             passes=(),
                             schedule="asap")
    assert "pulse.cosine" in compiled.mlir
    assert "pulse.custom_samples" in compiled.mlir
    assert "pulse.add" in compiled.mlir
    assert "pulse.scale" in compiled.mlir
    assert "pulse.neg" in compiled.mlir
    assert _drive_timings(compiled.mlir) == [(0, 4)]


def test_custom_waveform_preserves_callback_symbol():

    @pulse.kernel
    def kernel():
        q = pulse.qudit_ref()
        line, tone = get_drive_line(q)
        drive(line, custom(8, "calibrated_envelope"), tone)

    compiled = pulse.compile(kernel, [], qubit_freq_hz={0: 5.0e9}, passes=())
    assert "@calibrated_envelope" in compiled.mlir


def test_concrete_numeric_arguments_are_not_made_symbolic():

    @pulse.kernel
    def kernel(duration, amplitude):
        q = pulse.qudit_ref()
        line, tone = get_drive_line(q)
        drive(line, gaussian(duration, amplitude, 5.0), tone)

    compiled = pulse.compile(kernel, [20, 0.4], qubit_freq_hz={0: 5.0e9})
    assert not compiled.is_parametric
    assert _drive_timings(compiled.mlir) == [(0, 20)]


def test_symbolic_type_is_inferred_from_use_not_name():

    @pulse.kernel
    def kernel(q, tau):
        line, _ = get_drive_line(q)
        wait(line, tau)

    compiled = pulse.compile(kernel, [pulse.qudit_ref()],
                             qubit_freq_hz={0: 5.0e9})
    assert "%arg0: i64" in compiled.mlir
    assert compiled(tau=17).mlir.count("duration_vtu = 17") == 1


def test_symbolic_arithmetic_and_explicit_cast_are_lowered():

    @pulse.kernel
    def kernel(q, sigma: float, delay: int):
        line, tone = get_drive_line(q)
        waveform = gaussian(int(4 * sigma), 0.2, sigma)
        drive(line, waveform, tone)
        wait(line, delay * 2 + 1)

    compiled = pulse.compile(kernel, [pulse.qudit_ref()],
                             qubit_freq_hz={0: 5.0e9},
                             schedule="asap")
    assert "arith.mulf" in compiled.mlir
    assert "arith.fptosi" in compiled.mlir
    specialized = compiled(sigma=5.0, delay=7)
    assert _drive_timings(specialized.mlir) == [(0, 20)]
    assert "duration_vtu = 15" in specialized.mlir


def test_unknown_pass_is_rejected():

    @pulse.kernel
    def kernel():
        pass

    with pytest.raises(ValueError, match="Unknown pulse passes"):
        pulse.compile(kernel, [], passes=("typo",))


def test_asap_and_alap_have_distinct_correct_placement():

    @pulse.kernel
    def kernel(q0, q1):
        line0, tone0 = get_drive_line(q0)
        line1, tone1 = get_drive_line(q1)
        drive(line0, square(40, 0.2), tone0)
        drive(line1, square(100, 0.2), tone1)

    args = [pulse.qudit_ref(), pulse.qudit_ref()]
    frequencies = {0: 5.0e9, 1: 5.1e9}
    asap = pulse.compile(kernel,
                         args,
                         qubit_freq_hz=frequencies,
                         schedule="asap",
                         passes=())
    alap = pulse.compile(kernel,
                         args,
                         qubit_freq_hz=frequencies,
                         schedule="alap",
                         passes=())
    assert _drive_timings(asap.mlir) == [(0, 40), (0, 100)]
    assert _drive_timings(alap.mlir) == [(60, 40), (0, 100)]


def test_compile_uses_resource_machine_for_overlapping_intervals():

    @pulse.kernel
    def kernel(q0, q1):
        line0, tone0 = get_drive_line(q0)
        line1, tone1 = get_drive_line(q1)
        drive(line0, square(100, 0.2), tone0)
        drive(line1, square(40, 0.2), tone1)

    compiled = pulse.compile(
        kernel, [pulse.qudit_ref(), pulse.qudit_ref()],
        qubit_freq_hz={
            0: 5.0e9,
            1: 5.1e9
        },
        schedule="rcp",
        passes=(),
        machine=MachineModel(max_concurrent_drives=1))
    assert _drive_timings(compiled.mlir) == [(0, 100), (100, 40)]


def test_python_scheduler_tracks_line_lineage_and_resource_intervals():
    builder = ProgramBuilder("lineage")
    line, tone = builder.get_drive_line(0, 5.0e9)
    for duration in (10, 20, 30):
        waveform = builder.square(duration, 0.2)
        line, tone = builder.drive(line, waveform, tone)
    program = builder.build()
    events, metrics = schedule_asap(program)
    drives = [event for event in events if event.kind == "drive"]
    assert [event.start_vtu for event in drives] == [0, 10, 30]
    assert metrics.total_length_vtu == 60

    independent = ProgramBuilder("resources")
    for qubit in range(3):
        line, tone = independent.get_drive_line(qubit, 5.0e9 + qubit * 1.0e8)
        waveform = independent.square(100, 0.2)
        independent.drive(line, waveform, tone)
    events, _ = schedule_rcp(independent.build(),
                             MachineModel(max_concurrent_drives=2))
    starts = [event.start_vtu for event in events if event.kind == "drive"]
    assert starts == [0, 0, 100]


def test_python_alap_delays_independent_shorter_operation():
    builder = ProgramBuilder("alap")
    line0, tone0 = builder.get_drive_line(0, 5.0e9)
    line1, tone1 = builder.get_drive_line(1, 5.1e9)
    builder.drive(line0, builder.square(40, 0.2), tone0)
    builder.drive(line1, builder.square(100, 0.2), tone1)
    events, metrics = schedule_alap(builder.build())
    drives = [event for event in events if event.kind == "drive"]
    assert [event.start_vtu for event in drives] == [60, 0]
    assert metrics.total_length_vtu == 100


def test_packed_decoder_rejects_truncated_and_unknown_records():
    builder = PulseModuleBuilder()
    frequencies = np.array([], dtype=np.float64)
    with pytest.raises(Exception, match="truncated record"):
        builder.build_from_packed(np.array([3 | (4 << 8)], dtype=np.int64), 2.0,
                                  0, frequencies)
    with pytest.raises(Exception, match="unknown opcode"):
        builder.build_from_packed(np.array([255], dtype=np.int64), 2.0, 0,
                                  frequencies)


def test_qudit_ref_can_bind_a_physical_target_index():

    @pulse.kernel
    def kernel(qubit):
        line, _tone = get_drive_line(qubit)
        wait(line, 10)

    compiled = pulse.compile(kernel, [pulse.qudit_ref(4)],
                             qubit_freq_hz={4: 5.2e9},
                             passes=())
    assert "qubit = 4 : i64" in compiled.mlir
    assert "frequency_hz = 5.200000e+09 : f64" in compiled.mlir

    with pytest.raises(ValueError, match="non-negative"):
        pulse.qudit_ref(-1)


def test_virtual_time_arguments_are_not_silently_truncated():

    @pulse.kernel
    def kernel(qubit):
        line, tone = get_drive_line(qubit)
        drive(line, square(10.5, 0.2), tone)

    with pytest.raises(CompilationError, match="must be an integer"):
        pulse.compile(kernel, [pulse.qudit_ref()], qubit_freq_hz={0: 5.0e9})
