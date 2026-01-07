# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, os, pytest
from cudaq import spin, Schedule, RungeKuttaIntegrator
import numpy as np

skipIfUnsupported = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 1 and cudaq.has_target('dynamics')),
    reason="dynamics backend not available or not a multi-GPU machine")


@pytest.fixture(autouse=True)
def do_something():
    cudaq.mpi.initialize()
    cudaq.set_target('dynamics')
    yield
    cudaq.reset_target()
    cudaq.mpi.finalize()


@skipIfUnsupported
def testMpiRun():
    # Large number of spins
    N = 20
    dimensions = {}
    for i in range(N):
        dimensions[i] = 2

    # Observable is the average magnetization operator
    avg_magnetization_op = spin.empty()
    for i in range(N):
        avg_magnetization_op += (spin.z(i) / N)

    # Arbitrary coupling constant
    g = 1.0
    # Construct the Hamiltonian
    H = spin.empty()
    for i in range(N):
        H += 2 * np.pi * spin.x(i)
        H += 2 * np.pi * spin.y(i)
    for i in range(N - 1):
        H += 2 * np.pi * g * spin.x(i) * spin.x(i + 1)
        H += 2 * np.pi * g * spin.y(i) * spin.z(i + 1)

    steps = np.linspace(0.0, 1, 100)
    schedule = Schedule(steps, ["time"])

    # Initial state (expressed as an enum)
    psi0 = cudaq.dynamics.InitialState.ZERO

    # Run the simulation
    evolution_result = cudaq.evolve(
        H,
        dimensions,
        schedule,
        psi0,
        observables=[avg_magnetization_op],
        collapse_operators=[],
        store_intermediate_results=cudaq.IntermediateResultSave.NONE,
        integrator=RungeKuttaIntegrator())


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
