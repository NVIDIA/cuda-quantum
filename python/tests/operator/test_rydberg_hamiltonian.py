# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json
import pytest
from cudaq.dynamics import Schedule
from cudaq.operators import RydbergHamiltonian, ScalarOperator


def _scalars():
    zero = ScalarOperator.const(0.0)
    return zero, zero, zero


def test_atom_filling_defaults_to_all_filled():
    sites = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
    amp, phase, delta = _scalars()
    h = RydbergHamiltonian(atom_sites=sites,
                           amplitude=amp,
                           phase=phase,
                           delta_global=delta)
    assert h.atom_filling == [1, 1, 1]


def test_atom_filling_none_is_treated_as_unprovided():
    # `atom_filling` is Optional and the docstring promises that when it is
    # not provided all sites are filled; an explicit None must behave the same
    # rather than raising `TypeError: object of type 'NoneType' has no len()`.
    sites = [(0.0, 0.0), (0.0, 1.0)]
    amp, phase, delta = _scalars()
    h = RydbergHamiltonian(atom_sites=sites,
                           amplitude=amp,
                           phase=phase,
                           delta_global=delta,
                           atom_filling=None)
    assert h.atom_filling == [1, 1]


def test_empty_atom_filling_is_treated_as_unprovided():
    # An explicitly empty `atom_filling` carries no per-site information, so it
    # must behave like an unprovided value and default to all sites filled.
    sites = [(0.0, 0.0), (0.0, 1.0)]
    amp, phase, delta = _scalars()
    h = RydbergHamiltonian(atom_sites=sites,
                           amplitude=amp,
                           phase=phase,
                           delta_global=delta,
                           atom_filling=[])
    assert h.atom_filling == [1, 1]


def test_mismatched_atom_filling_still_rejected():
    sites = [(0.0, 0.0), (0.0, 1.0)]
    amp, phase, delta = _scalars()
    with pytest.raises(ValueError, match="must be equal"):
        RydbergHamiltonian(atom_sites=sites,
                           amplitude=amp,
                           phase=phase,
                           delta_global=delta,
                           atom_filling=[1])


def _capture_analog_programs(monkeypatch):
    from cudaq.dynamics import evolution
    programs = []
    monkeypatch.setattr(
        evolution.cudaq_runtime, "launch_analog_kernel",
        lambda name, program, shots: programs.append(json.loads(program)))
    return evolution, programs


def test_analog_launch_reuses_schedule(monkeypatch):
    # Reusing a Schedule must send the same time points every launch.
    evolution, programs = _capture_analog_programs(monkeypatch)
    h = RydbergHamiltonian(atom_sites=[(0.0, 0.0), (1e-4, 0.0)],
                           amplitude=ScalarOperator.const(1e7),
                           phase=ScalarOperator.const(0.0),
                           delta_global=ScalarOperator.const(0.0))
    schedule = Schedule([0.0, 1e-7], ["t"])
    for _ in range(2):
        evolution._launch_analog_hamiltonian_kernel("pasqal", h, schedule, 100)
    assert len(programs) == 2
    assert programs[0] == programs[1]
    amplitude = programs[0]["hamiltonian"]["drivingFields"][0]["amplitude"]
    assert len(amplitude["time_series"]["times"]) == 2


def test_analog_launch_rejects_empty_schedule(monkeypatch):
    evolution, programs = _capture_analog_programs(monkeypatch)
    amp, phase, delta = _scalars()
    h = RydbergHamiltonian(atom_sites=[(0.0, 0.0)],
                           amplitude=amp,
                           phase=phase,
                           delta_global=delta)
    with pytest.raises(ValueError, match="at least one time step"):
        evolution._launch_analog_hamiltonian_kernel("pasqal", h,
                                                    Schedule([], ["t"]), 100)
    assert programs == []
