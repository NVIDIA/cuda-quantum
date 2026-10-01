# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
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


def test_mismatched_atom_filling_still_rejected():
    sites = [(0.0, 0.0), (0.0, 1.0)]
    amp, phase, delta = _scalars()
    with pytest.raises(ValueError, match="must be equal"):
        RydbergHamiltonian(atom_sites=sites,
                           amplitude=amp,
                           phase=phase,
                           delta_global=delta,
                           atom_filling=[1])
