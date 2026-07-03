# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the program_to_pulse_mlir emitter."""

import math
import re

import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir


def test_simple_drive(simple_program):
    mlir = program_to_pulse_mlir(simple_program)
    assert "module @" in mlir
    assert "func.func @main()" in mlir
    assert "pulse.qudit_alloc" in mlir
    assert "pulse.get_drive_line" in mlir
    assert "pulse.gaussian" in mlir
    assert "pulse.drive" in mlir
    assert "return" in mlir


def test_two_qubit_sync(two_qubit_program):
    mlir = program_to_pulse_mlir(two_qubit_program)
    assert mlir.count("pulse.qudit_alloc") == 2
    assert mlir.count("pulse.get_drive_line") == 2
    assert "pulse.sync" in mlir
    assert len(re.findall(r"= pulse\.drive ", mlir)) == 2


def test_loop_program(echo_program):
    mlir = program_to_pulse_mlir(echo_program)
    assert "scf.for" in mlir
    assert "iter_args" in mlir
    assert "scf.yield" in mlir
    assert "arith.constant 0 : index" in mlir
    assert "arith.constant 5 : index" in mlir


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
    assert re.search(r"pulse\.square\s+20,\s*\[", mlir)


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
            assert ssa_name in defined, (
                f"SSA value {ssa_name} used before definition in: {line}")


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
