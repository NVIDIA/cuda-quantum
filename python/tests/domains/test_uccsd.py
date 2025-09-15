# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import importlib
import os
import shutil
from pathlib import Path

import cudaq
import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Ref: https://github.com/NVIDIA/cuda-quantum/issues/1954
    """
    import cudaq

    if not cudaq.globalAstRegistry:
        print("AST registry empty, patching uccsd.")
        # Now we make a copy for running the tests
        current_dir = Path(__file__).parent  # tests/domains directory
        src_file = current_dir.parent.parent / "cudaq" / "kernels" / "uccsd.py"

        if not src_file.exists():
            pytest.skip(
                f"Source file {src_file} not found. Skipping `uccsd` tests.")

        temp_file = Path("/tmp/fresh_uccsd.py")
        shutil.copy(src_file, temp_file)

        # Import the module from the temporary file
        spec = importlib.util.spec_from_file_location("fresh_uccsd", temp_file)
        fresh_uccsd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh_uccsd)

        # Save references to original module functions if they exist
        import cudaq.kernels
        original_uccsd = getattr(cudaq.kernels, "uccsd", None)
        original_num_params = getattr(cudaq.kernels, "uccsd_num_parameters",
                                      None)

        # Patch our fresh functions into cudaq.kernels
        cudaq.kernels.uccsd = fresh_uccsd.uccsd
        cudaq.kernels.uccsd_num_parameters = fresh_uccsd.uccsd_num_parameters

        # Run the tests
        yield

        # Clean up - remove the temporary file
        if temp_file.exists():
            os.unlink(temp_file)

        # Restore the original module functions
        cudaq.kernels.uccsd = original_uccsd
        cudaq.kernels.uccsd_num_parameters = original_num_params

    else:
        # When running the tests standalone, this is the path taken.
        print("AST registry not empty, skipping uccsd patching.")
        yield


def test_uccsd_builder():
    from cudaq.kernels import uccsd

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


def test_uccsd_sample():

    from cudaq.kernels import uccsd

    num_electrons = 2
    num_qubits = 8

    thetas = [
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558,
        -0.00037043841404585794, 0.0003811110195084151, 0.2286823796532558
    ]

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(num_qubits)
        for i in range(num_electrons):
            x(qubits[i])
        uccsd(qubits, thetas, num_electrons, num_qubits)

    counts = cudaq.sample(kernel, shots_count=1000)
    assert len(counts) == 6
    assert '00000011' in counts
    assert '00000110' in counts
    assert '00010010' in counts
    assert '01000010' in counts
    assert '10000001' in counts
    assert '11000000' in counts
