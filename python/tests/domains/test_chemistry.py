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

openfermion_pyscf = pytest.importorskip('openfermionpyscf')


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
    energy, params = cudaq.vqe(kernel,
                               molecule,
                               optimizer,
                               parameter_count=num_parameters)
    print(energy, params)
    assert np.isclose(-1.137, energy, rtol=1e-3)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
