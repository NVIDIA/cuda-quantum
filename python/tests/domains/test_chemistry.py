# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import numpy as np

import cudaq

openfermion_pyscf = pytest.importorskip('openfermionpyscf')

@pytest.fixture(autouse=True)
def do_something():
    if os.getenv("CUDAQ_PYTEST_EAGER_MODE") == 'ON':
        cudaq.disable_jit()
    yield
    if cudaq.is_jit_enabled():
        cudaq.__clearKernelRegistries()
    cudaq.enable_jit()

def test_HamiltonianGenH2Sto3g():

    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, 'sto-3g', 1, 0)
    energy = molecule.to_matrix().minimal_eigenvalue()
    assert np.isclose(energy, -1.137, rtol=1e-3)


def test_HamiltonianGenH2631g():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, '6-31g', 1, 0)
    energy = molecule.to_matrix().minimal_eigenvalue()
    assert np.isclose(energy, -1.1516, rtol=1e-3)


def testUCCSD():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, 'sto-3g', 1, 0)

    # Get the number of fermions and orbitals / qubits
    numElectrons = data.n_electrons
    numQubits = 2 * data.n_orbitals

    # create the ansatz
    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(4)
    # hartree fock
    kernel.x(qubits[0])
    kernel.x(qubits[1])
    cudaq.kernels.uccsd(kernel, qubits, thetas, numElectrons, numQubits)

    num_parameters = cudaq.kernels.uccsd_num_parameters(numElectrons, numQubits)

    # Run VQE
    optimizer = cudaq.optimizers.COBYLA()
    energy, params = cudaq.vqe(kernel,
                               molecule,
                               optimizer,
                               parameter_count=num_parameters)
    print(energy, params)
    assert np.isclose(-1.137, energy, rtol=1e-3)


def testHWE():
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, 'sto-3g', 1, 0)

    # Get the number of qubits
    numQubits = 2 * data.n_orbitals

    # select number of repeating layers in ansatz
    numLayers = 4

    # create the ansatz
    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(numQubits)
    # hartree fock
    kernel.x(qubits[0])
    kernel.x(qubits[1])
    cudaq.kernels.hwe(kernel, qubits, numQubits, numLayers, thetas)

    num_parameters = cudaq.kernels.num_hwe_parameters(numQubits, numLayers)
    assert np.equal(40, num_parameters)

    # Run VQE
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 500
    energy, params = cudaq.vqe(kernel,
                               molecule,
                               optimizer,
                               parameter_count=num_parameters)
    print(energy, params)
    assert np.isclose(-1.13, energy, rtol=1e-2)

def test_excitations():
    
    from cudaq.kernels import test_excitations

    # num electrons = 2, num qubits = 4
    assert cudaq.kernels.test_excitations(2, 4, 0, 0, 0) == 0
    assert cudaq.kernels.test_excitations(2, 4, 0, 1, 0) == 2
    assert cudaq.kernels.test_excitations(2, 4, 0, 0, 1) == 1
    assert cudaq.kernels.test_excitations(2, 4, 0, 1, 1) == 3
    assert cudaq.kernels.test_excitations(2, 4, 0, 0, 2) == 0
    assert cudaq.kernels.test_excitations(2, 4, 0, 1, 2) == 1
    assert cudaq.kernels.test_excitations(2, 4, 0, 2, 2) == 3
    assert cudaq.kernels.test_excitations(2, 4, 0, 3, 2) == 2

    assert cudaq.kernels.test_excitations(4, 8, 0, 0, 0) == 0 
    assert cudaq.kernels.test_excitations(4, 8, 0, 1, 0) == 4
    assert cudaq.kernels.test_excitations(4, 8, 1, 0, 0) == 0
    assert cudaq.kernels.test_excitations(4, 8, 1, 1, 0) == 6
    assert cudaq.kernels.test_excitations(4, 8, 2, 0, 0) == 2 
    assert cudaq.kernels.test_excitations(4, 8, 2, 1, 0) == 4
    assert cudaq.kernels.test_excitations(4, 8, 3, 0, 0) == 2 
    assert cudaq.kernels.test_excitations(4, 8, 3, 1, 0) == 6

    assert cudaq.kernels.test_excitations(4, 8, 0, 0, 1) == 1
    assert cudaq.kernels.test_excitations(4, 8, 0, 1, 1) == 5
    assert cudaq.kernels.test_excitations(4, 8, 1, 0, 1) == 1
    assert cudaq.kernels.test_excitations(4, 8, 1, 1, 1) == 7
    assert cudaq.kernels.test_excitations(4, 8, 2, 0, 1) == 3
    assert cudaq.kernels.test_excitations(4, 8, 2, 1, 1) == 5
    assert cudaq.kernels.test_excitations(4, 8, 3, 0, 1) == 3 
    assert cudaq.kernels.test_excitations(4, 8, 3, 1, 1) == 7

    assert cudaq.kernels.test_excitations(4, 8, 0, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 0, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 0, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 0, 3, 2) == 4
    
    assert cudaq.kernels.test_excitations(4, 8, 1, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 1, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 1, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 1, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 2, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 2, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 2, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 2, 3, 2) == 4

    assert cudaq.kernels.test_excitations(4, 8, 3, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 3, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 3, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 3, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 4, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 4, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 4, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 4, 3, 2) == 4

    assert cudaq.kernels.test_excitations(4, 8, 5, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 5, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 5, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 5, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 6, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 6, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 6, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 6, 3, 2) == 4
        
    assert cudaq.kernels.test_excitations(4, 8, 7, 0, 2) == 0
    assert cudaq.kernels.test_excitations(4, 8, 7, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 7, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 7, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 8, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 8, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 8, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 8, 3, 2) == 4

    assert cudaq.kernels.test_excitations(4, 8, 9, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 9, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 9, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 9, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 10, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 10, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 10, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 10, 3, 2) == 4
        
    assert cudaq.kernels.test_excitations(4, 8, 11, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 11, 1, 2) == 1
    assert cudaq.kernels.test_excitations(4, 8, 11, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 11, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 12, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 12, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 12, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 12, 3, 2) == 4

    assert cudaq.kernels.test_excitations(4, 8, 13, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 13, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 13, 2, 2) == 5
    assert cudaq.kernels.test_excitations(4, 8, 13, 3, 2) == 6

    assert cudaq.kernels.test_excitations(4, 8, 14, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 14, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 14, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 14, 3, 2) == 4
        
    assert cudaq.kernels.test_excitations(4, 8, 15, 0, 2) == 2
    assert cudaq.kernels.test_excitations(4, 8, 15, 1, 2) == 3
    assert cudaq.kernels.test_excitations(4, 8, 15, 2, 2) == 7
    assert cudaq.kernels.test_excitations(4, 8, 15, 3, 2) == 6


def test_uccsd_kernel():

    @cudaq.kernel 
    def ansatz(thetas : list[float]):#, numElectrons:int, numQubits:int):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        cudaq.kernels.uccsd(q, thetas, 2, 4)
    
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, 'sto-3g', 1, 0)

    # Get the number of fermions and orbitals / qubits
    numElectrons = data.n_electrons
    numQubits = 2 * data.n_orbitals
    num_parameters = cudaq.kernels.uccsd_num_parameters(numElectrons, numQubits)

    xInit = [0.0]*num_parameters

    print(cudaq.observe(ansatz, molecule, xInit).expectation())

    optimizer = cudaq.optimizers.COBYLA()
    energy, params = cudaq.vqe(ansatz,
                               molecule,
                               optimizer,
                               parameter_count=num_parameters)
    print(energy, params)

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
