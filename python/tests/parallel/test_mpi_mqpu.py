# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, os, pytest
from cudaq import spin
import numpy as np

skipIfUnsupported = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-mqpu')),
    reason="nvidia-mqpu backend not available or mpi not found")


@pytest.fixture(scope='session', autouse=True)
def mpi_init_finalize():
    cudaq.mpi.initialize()
    yield
    cudaq.mpi.finalize()


@pytest.fixture(autouse=True)
def do_something():
    cudaq.set_target('nvidia-mqpu')
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def check_mpi(entity):
    target = cudaq.get_target()
    numQpus = target.num_qpus()
    if numQpus == 0:
        pytest.skip("No QPUs available for target, skipping MPI test")
    else:
        print(
            f"Target: {target}, NumQPUs: {numQpus}, MPI Ranks: {cudaq.mpi.num_ranks()}"
        )
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    # Confirmed expectation value for this system when `theta=0.59`.
    want_expectation_value = -1.7487948611472093

    # Get the `cudaq.ObserveResult` back from `cudaq.observe()`.
    # No shots provided.
    result_no_shots = cudaq.observe(entity,
                                    hamiltonian,
                                    0.59,
                                    execution=cudaq.parallel.mpi)
    expectation_value_no_shots = result_no_shots.expectation()
    assert np.isclose(want_expectation_value, expectation_value_no_shots)

    # Test all gather
    numRanks = cudaq.mpi.num_ranks()
    local = [1.0]
    globalList = cudaq.mpi.all_gather(numRanks, local)
    assert len(globalList) == numRanks


@skipIfUnsupported
def testMPI():

    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])

    check_mpi(kernel)


@skipIfUnsupported
def testMPI_kernel():

    @cudaq.kernel
    def kernel(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    check_mpi(kernel)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
