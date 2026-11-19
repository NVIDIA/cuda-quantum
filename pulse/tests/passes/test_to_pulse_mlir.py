# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the program_to_pulse_mlir emitter."""

import math
import re

import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.targets.base import Qubit, Target


def test_simple_drive(simple_program):
    mlir = program_to_pulse_mlir(simple_program)
    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.qudit_alloc" in mlir
    assert "pulse.get_drive_line" in mlir
    assert "pulse.gaussian" in mlir
    assert "pulse.drive" in mlir
    assert "return" in mlir


def test_target_and_evolution_metadata(simple_program):
    target = Target(
        name="sim",
        qubits={
            0:
                Qubit(
                    index=0,
                    frequency_hz=5.0e9,
                    anharmonicity_hz=-200.0e6,
                    t1_us=50.0,
                    t2_star_us=30.0,
                    drive_params={"amplitude_scale_rad_per_ns": 0.25},
                )
        },
    )
    mlir = program_to_pulse_mlir(simple_program,
                                 target=target,
                                 t_start=1.0,
                                 t_end=20.0,
                                 num_steps=64,
                                 integrator="rk4")
    assert "qop.t_start = 1.000000000000000e+00 : f64" in mlir
    assert "qop.t_end = 2.000000000000000e+01 : f64" in mlir
    assert "qop.num_steps = 64 : i64" in mlir
    assert 'qop.integrator = "rk4"' in mlir
    assert "pulse.t1_times = [5.000000000000000e+04 : f64]" in mlir
    assert "pulse.drive_scale_rad_per_ns = array<f64: " in mlir


def test_two_qubit_sync(two_qubit_program):
    mlir = program_to_pulse_mlir(two_qubit_program)
    assert mlir.count("pulse.qudit_alloc") == 2
    assert mlir.count("pulse.get_drive_line") == 2
    assert "pulse.sync" in mlir
    assert len(re.findall(r"= pulse\.drive ", mlir)) == 2


def test_loop_program(echo_program):
    mlir = program_to_pulse_mlir(echo_program)
    assert len(re.findall(r"= pulse\.drive ", mlir)) == 10
    assert "scf.for" not in mlir


def test_shift_phase_emission():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        shift_phase(t0, math.pi / 4)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.shift_phase" in mlir
    assert re.search(r"arith\.constant\s+7\.8539\d+e-01\s*:\s*f64", mlir)


def test_set_phase_emission():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        set_phase(t0, 1.0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.set_phase" in mlir


def test_wait_emission():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)
        wait(d0, 50)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.wait" in mlir
    assert "pulse.duration_from_int" in mlir
    assert "arith.constant 50 : i64" in mlir


def test_readout_emission():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        r0, rt0 = get_readout_line(q0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)
        rwf = square(20, 0.1)
        readout(r0, rwf, rt0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.get_readout_line" in mlir
    assert "pulse.readout" in mlir
    assert "pulse.square" in mlir
    assert '"iq"' in mlir


def test_square_waveform_iq_pair():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = square(20, 0.1)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert re.search(r"pulse\.square\s+%dur\d+,\s*%amp\d+,\s*%amp\d+", mlir)
    assert ": i64, f64, f64 -> !pulse.waveform" in mlir


def test_drag_waveform():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = drag(40, 0.3, 10.0, 0.5)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.drag" in mlir


def test_gaussian_square_flat_width_becomes_edge_duration():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian_square(100, 0.3, 10.0, 20)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "arith.constant 40 : i64" in mlir
    assert "pulse.gaussian_square" in mlir


def test_gaussian_square_rejects_invalid_flat_width():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian_square(100, 0.3, 10.0, 100)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    with pytest.raises(ValueError, match="width must satisfy"):
        program_to_pulse_mlir(prog)


def test_ssa_threading_correctness(two_qubit_program):
    """SSA values must never be used before they're defined."""
    mlir = program_to_pulse_mlir(two_qubit_program)
    defined = set()
    for line in mlir.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line in ("{", "}"):
            continue
        lhs_match = re.match(r"^((?:%\w+,?\s*)+)\s*=", line)
        if lhs_match:
            for m in re.finditer(r"%(\w+)", lhs_match.group(1)):
                defined.add(m.group(0))
        if "=" in line:
            rhs = line.split("=", 1)[1]
        else:
            rhs = line
        for m in re.finditer(r"%(\w+)", rhs):
            ssa_name = m.group(0)
            if ssa_name.startswith("%arg") or ssa_name.startswith("%iv"):
                continue
            assert ssa_name in defined, f"SSA value {ssa_name} used before definition in: {line}"


def test_module_name(simple_program):
    """Module name should come from the program name."""
    mlir = program_to_pulse_mlir(simple_program)
    assert f"module @{simple_program.name}" in mlir


def test_scheduling_attrs_preserved():

    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)

    ir = k(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
    mlir = program_to_pulse_mlir(prog)
    assert "duration_vtu = 40" in mlir
