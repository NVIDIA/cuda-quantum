# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq.dynamics import Schedule
from cudaq.operators import RydbergHamiltonian, ScalarOperator
import numpy as np
import os
import pytest

# The `pasqal` target only exists when CUDA-Q was configured with the Pasqal
# backend enabled (CUDAQ_ENABLE_PASQAL_BACKEND). Guard on what the build
# actually produced -- the session fixture below calls set_target during
# setup, which raises on a build that correctly does not provide it.
pytestmark = pytest.mark.skipif(
    not cudaq.has_target("pasqal"),
    reason="Could not find `pasqal` in installation")
skipIfPasqalEmulationUnavailable = pytest.mark.skipif(
    not (cudaq.has_target("pasqal") and cudaq.has_target("dynamics") and
         cudaq.num_available_gpus() > 0),
    reason='Pasqal emulation requires the dynamics backend and a CUDA GPU')


@pytest.fixture(scope="session", autouse=True)
def set_up_target():
    # NOTE: Credentials can be set with environment variables.
    # This test covers the direct `pasqal` backend only.
    # QRMI-routed execution is validated separately because it requires a
    # supported QRMI build and a compatible cluster resource manager.
    cudaq.set_target("pasqal")
    yield "Running the tests."
    cudaq.reset_target()


@skipIfPasqalEmulationUnavailable
def test_emulate_phase_sequence():
    """Resolve the phase sign with a detuned two-pulse experiment."""
    from scipy.linalg import expm

    cudaq.set_target("pasqal", emulate=True)
    cudaq.set_random_seed(21)
    x = np.array([[0., 1.], [1., 0.]])
    y = np.array([[0., -1j], [1j, 0.]])
    number = np.diag([0., 1.])
    first = 4. * x - 4. * number
    second = 4. * y - 4. * number
    state = expm(-1j * second * 0.2) @ expm(-1j * first * 0.2) @ [1., 0.]
    result = cudaq.evolve(RydbergHamiltonian(
        atom_sites=[(0., 0.)],
        amplitude=ScalarOperator.const(8e6),
        phase=ScalarOperator(lambda t: 0. if t.real < 2e-7 else np.pi / 2),
        delta_global=ScalarOperator.const(4e6)),
                          schedule=Schedule([0., 2e-7, 4e-7], ["t"]),
                          shots_count=10000)
    assert result.probability("1") == pytest.approx(abs(state[1])**2, abs=0.025)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
