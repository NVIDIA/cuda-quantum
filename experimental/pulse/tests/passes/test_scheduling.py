# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudaq_pulse.passes.ir_types import Program
from cudaq_pulse.passes.scheduling import (
    schedule_asap,
    schedule_alap,
    schedule_rcp,
    MachineModel,
)


def test_asap_simple(simple_program):
    events, metrics = schedule_asap(simple_program)
    assert metrics.total_length_vtu > 0
    assert metrics.op_count > 0


def test_alap_simple(simple_program):
    events, metrics = schedule_alap(simple_program)
    assert metrics.total_length_vtu > 0


def test_asap_two_qubit(two_qubit_program):
    events, metrics = schedule_asap(two_qubit_program)
    assert metrics.total_length_vtu > 0


def test_alap_two_qubit(two_qubit_program):
    events, metrics = schedule_alap(two_qubit_program)
    assert metrics.total_length_vtu > 0


def test_rcp_two_qubit(two_qubit_program):
    machine = MachineModel(max_concurrent_drives=2, max_concurrent_readouts=2)
    events, metrics = schedule_rcp(two_qubit_program, machine)
    assert metrics.total_length_vtu > 0


def test_asap_alap_same_makespan(simple_program):
    _, asap_m = schedule_asap(simple_program)
    _, alap_m = schedule_alap(simple_program)
    assert asap_m.total_length_vtu == alap_m.total_length_vtu


def test_echo_scheduling(echo_program):
    events, metrics = schedule_asap(echo_program)
    assert metrics.total_length_vtu > 0
    assert metrics.op_count >= 6  # loop body ops + loop structure
