# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, os, pytest, random, timeit
from cudaq import spin
import numpy as np

skipIfUnsupported = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.mpi.is_initialized() and cudaq.has_target('nvidia-mqpu')),
    reason="nvidia-mqpu backend not available or mpi not found"
)


@skipIfUnsupported
def testMPI():
    cudaq.set_target('nvidia-mqpu')
    cudaq.mpi.initialize()

    target = cudaq.get_target()
    numQpus = target.num_qpus()

    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    # Confirmed expectation value for this system when `theta=0.59`.
    want_expectation_value = -1.7487948611472093

    # Get the `cudaq.ObserveResult` back from `cudaq.observe()`.
    # No shots provided.
    result_no_shots = cudaq.observe(kernel,
                                    hamiltonian,
                                    0.59,
                                    execution=cudaq.parallel.mpi)
    expectation_value_no_shots = result_no_shots.expectation_z()
    assert np.isclose(want_expectation_value, expectation_value_no_shots)

    # Test all gather 
    numRanks = cudaq.mpi.num_ranks()
    local = [1.0]
    globalList = cudaq.mpi.all_gather(numRanks, local)
    assert len(globalList) == numRanks

    cudaq.reset_target()
    cudaq.mpi.finalize()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
