# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""GPU integration tests: T1 decoherence during idle (wait) periods.

Ported from ``qpu/physics/test_idle_evolution.cpp``. Verifies that free
evolution under a T1 Lindblad term follows P(|1>, t) = P0 * exp(-t/T1), that
short idles cause negligible decay, and that splitting an idle into several
shorter waits yields the same decay (schedule/integrator consistency).
"""

import math

import numpy as np
import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir
from cudaq_pulse.passes import run_canonicalize, run_virtual_z, run_fusion, schedule_alap
from cudaq_pulse.targets import Qubit, Target


def _gpu_available():
    try:
        from cudaq_pulse.runtime.jit import _check_gpu_available

        return _check_gpu_available()
    except Exception:
        return False


gpu = pytest.mark.gpu
requires_gpu = pytest.mark.skipif(not _gpu_available(),
                                  reason="No GPU/cuDensityMat")

# T1 chosen large relative to the 20 ns preparation pulse so in-pulse decay is
# negligible and the idle-period decay cleanly follows exp(-t/T1).
_T1_US = 2.0
_T1_NS = _T1_US * 1.0e3
_CLOCK_GHZ = 2.0  # 2 virtual time units per nanosecond


def _target():
    return Target(
        name="idle-evolution-test",
        qubits={
            0:
                Qubit(index=0,
                      frequency_hz=5.0e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=_T1_US,
                      t2_star_us=0.0,
                      drive_params={"amplitude_scale_rad_per_ns": 1.0})
        },
    )


def _p1_after_wait(wait_ns):
    """Prepare |1> with a pi pulse, idle for wait_ns, return P(|1>)."""
    wait_vtu = int(round(wait_ns * _CLOCK_GHZ))

    if wait_vtu > 0:

        @pulse.kernel
        def kernel(q0):
            d0, t0 = get_drive_line(q0)
            drive(d0, square(40, math.pi / 20.0), t0)
            wait(d0, wait_vtu)
    else:

        @pulse.kernel
        def kernel(q0):
            d0, t0 = get_drive_line(q0)
            drive(d0, square(40, math.pi / 20.0), t0)

    t_end = 20.0 + wait_ns
    result = pulse.evolve(kernel(pulse.qudit_ref()),
                          target=_target(),
                          t_start=0.0,
                          t_end=t_end,
                          num_steps=max(1, int(round(t_end))),
                          integrator="rk4")
    return result.final_state[1, 1].real


def test_idle_evolution_mlir_structure():
    """A pi-pulse-plus-idle kernel lowers to valid MLIR with a wait op."""

    @pulse.kernel
    def kernel(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)
        wait(d0, 1000)

    prog = to_program(kernel(pulse.qudit_ref()),
                      clock_ghz=_CLOCK_GHZ,
                      qubit_freq_hz={0: 5.0e9})
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    schedule_alap(prog)
    mlir = program_to_pulse_mlir(prog)
    assert "pulse.wait" in mlir


@gpu
@requires_gpu
def test_t1_decay_follows_exponential():
    """P(|1>) at t = frac * T1 matches P0 * exp(-frac) across a sweep."""
    fractions = [0.0, 0.25, 0.5, 1.0]
    p1 = [_p1_after_wait(f * _T1_NS) for f in fractions]

    p0 = p1[0]
    assert p0 > 0.95  # pi pulse leaves the qubit excited
    for i in range(1, len(fractions)):
        assert p1[i] < p1[i - 1]  # monotonic decay
        expected = p0 * math.exp(-fractions[i])
        assert p1[i] == pytest.approx(expected, abs=0.1)
    assert p1[-1] < 0.5  # after one T1, well below half


@gpu
@requires_gpu
def test_short_idle_minimal_decay():
    """A very short idle (~1% of T1) barely perturbs the excited population."""
    p1_ref = _p1_after_wait(0.0)
    p1_short = _p1_after_wait(0.01 * _T1_NS)
    assert p1_short == pytest.approx(p1_ref, abs=0.03)


@gpu
@requires_gpu
def test_split_idle_matches_single_idle():
    """Splitting one idle into several shorter waits gives the same decay."""

    @pulse.kernel
    def single(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)
        wait(d0, 2000)  # 1000 ns in one wait

    @pulse.kernel
    def split(q0):
        d0, t0 = get_drive_line(q0)
        drive(d0, square(40, math.pi / 20.0), t0)
        wait(d0, 400)
        wait(d0, 400)
        wait(d0, 400)
        wait(d0, 400)
        wait(d0, 400)  # 5 x 200 ns == 1000 ns total

    def _run(kernel):
        result = pulse.evolve(kernel(pulse.qudit_ref()),
                              target=_target(),
                              t_start=0.0,
                              t_end=1020.0,
                              num_steps=1020,
                              integrator="rk4")
        return result.final_state[1, 1].real

    assert _run(single) == pytest.approx(_run(split), abs=0.02)
