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

    from cudaq.kernels import uccsd

    # create the ansatz
    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(4)
    # hartree fock
    kernel.x(qubits[0])
    kernel.x(qubits[1])
    kernel.apply_call(uccsd, qubits, thetas, numElectrons, numQubits)

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


def test_uccsd_kernel():

    from cudaq.kernels import uccsd

    @cudaq.kernel
    def ansatz(thetas: list[float]):  #, numElectrons:int, numQubits:int):
        q = cudaq.qvector(4)
        x(q[0])
        x(q[1])
        uccsd(q, thetas, 2, 4)

    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
    molecule, data = cudaq.chemistry.create_molecular_hamiltonian(
        geometry, 'sto-3g', 1, 0)

    # Get the number of fermions and orbitals / qubits
    numElectrons = data.n_electrons
    numQubits = 2 * data.n_orbitals
    num_parameters = cudaq.kernels.uccsd_num_parameters(numElectrons, numQubits)

    xInit = [0.0] * num_parameters

    print(cudaq.observe(ansatz, molecule, xInit).expectation())

    optimizer = cudaq.optimizers.COBYLA()
    energy, params = cudaq.vqe(ansatz,
                               molecule,
                               optimizer,
                               parameter_count=num_parameters)
    print(energy, params)


def test_doubles_alpha_bug():
    # AST Bridge was not loading pointers
    # when building lists, hence the following would
    # break at runtime

    n_occupied = 2
    n_virtual = 4
    occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
    virtual_alpha_indices = [i * 2 + 4 for i in range(n_virtual)]
    lenOccA = 2
    lenVirtA = 4
    nEle = 6

    @cudaq.kernel
    def test() -> bool:
        counter = 0
        doubles_a = [[0, 0, 0, 0] for k in range(nEle)]
        for p in range(lenOccA - 1):
            for q in range(p + 1, lenOccA):
                for r in range(lenVirtA - 1):
                    for s in range(r + 1, lenVirtA):
                        cudaq.dbg.ast.print_i64(p)
                        cudaq.dbg.ast.print_i64(q)
                        cudaq.dbg.ast.print_i64(r)
                        cudaq.dbg.ast.print_i64(s)
                        cudaq.dbg.ast.print_i64(occupied_alpha_indices[p])
                        cudaq.dbg.ast.print_i64(occupied_alpha_indices[q])
                        cudaq.dbg.ast.print_i64(virtual_alpha_indices[r])
                        cudaq.dbg.ast.print_i64(virtual_alpha_indices[s])
                        t = occupied_alpha_indices[p]
                        u = occupied_alpha_indices[q]
                        v = virtual_alpha_indices[r]
                        w = virtual_alpha_indices[s]
                        doubles_a[counter] = [
                            occupied_alpha_indices[p], u, v, w
                        ]
                        for i in range(4):
                            cudaq.dbg.ast.print_i64(doubles_a[counter][i])
                        counter = counter + 1
        return True

    # Test is that this compiles and runs successfully
    assert test()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
