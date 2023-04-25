# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq

def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


def test_HamiltonianGenH2Sto3g():
    geometry = [('H', (0.,0.,0.)), ('H', (0.,0.,.7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, 'sto-3g', 1, 0)
    energy = molecule.to_matrix().minimal_eigenvalue()
    assert_close(energy, -1.137, 1e-3)


def test_HamiltonianGenH2631g():
    geometry = [('H', (0.,0.,0.)), ('H', (0.,0.,.7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, '6-31g', 1, 0)
    energy = molecule.to_matrix().minimal_eigenvalue()
    assert_close(energy, -1.1516, 1e-3)  

# FIXME implement uccsd in python 
# def testUCCSD():
#     geometry = [('H', (0.,0.,0.)), ('H', (0.,0.,.7474))]
#     molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, 'sto-3g', 1, 0)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
