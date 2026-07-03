# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program


@pulse.kernel
def _simple_kernel(q0):
    d0, t0 = get_drive_line(q0)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


@pulse.kernel
def _two_qubit_kernel(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)
    sync(d0, d1)
    drive(d1, wf, t1)


@pulse.kernel
def _echo_kernel(q0):
    d0, t0 = get_drive_line(q0)
    for i in range(5):
        wf_pos = gaussian(40, 0.3, 10.0)
        drive(d0, wf_pos, t0)
        wait(d0, 20)
        wf_neg = gaussian(40, -0.3, 10.0)
        drive(d0, wf_neg, t0)


@pytest.fixture
def simple_program():
    """A minimal single-qubit drive program for testing."""
    ir = _simple_kernel(pulse.qudit_ref())
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})


@pytest.fixture
def two_qubit_program():
    """A two-qubit program with sync for testing."""
    ir = _two_qubit_kernel(pulse.qudit_ref(), pulse.qudit_ref())
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9, 1: 5.1e9})


@pytest.fixture
def echo_program():
    """The canonical echo program from the paper."""
    ir = _echo_kernel(pulse.qudit_ref())
    return _to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})
