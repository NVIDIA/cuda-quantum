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
from cudaq import spin

@pytest.fixture(autouse=True)
def do_something():
    if os.getenv("CUDAQ_PYTEST_EAGER_MODE") == 'OFF':
        cudaq.enable_jit()
    yield

    if cudaq.jit_enabled(): cudaq.__clearKernelRegistries()
    cudaq.disable_jit()

def test_synthesize():
    ## NOTE: Explicitly disable JIT for the next test
    cudaq.disable_jit()

    @cudaq.kernel
    def wontWork(numQubits: int):
        q = cudaq.qvector(numQubits)
        h(q)

    with pytest.raises(RuntimeError) as error:
        cudaq.synthesize(wontWork, 4)

    @cudaq.kernel(jit=True)
    def ghz(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubitIdx in enumerate(range(numQubits - 1)):
            x.ctrl(qubits[i], qubits[qubitIdx + 1])

    print(ghz)
    ghz_synth = cudaq.synthesize(ghz, 5)
    assert len(ghz_synth.argTypes) == 0

    counts = cudaq.sample(ghz_synth)
    counts.dump()
    assert len(counts) == 2 and '0' * 5 in counts and '1' * 5 in counts

    @cudaq.kernel(jit=True)
    def ansatz(angle: float):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle, q[1])
        x.ctrl(q[1], q[0])

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    ansatz_synth = cudaq.synthesize(ansatz, .59)
    result = cudaq.observe(ansatz_synth, hamiltonian)
    print(result.expectation())
    assert np.isclose(result.expectation(), -1.74, atol=1e-2)

    @cudaq.kernel(jit=True)
    def ansatzVec(angle: list[float]):
        q = cudaq.qvector(2)
        x(q[0])
        ry(angle[0], q[1])
        x.ctrl(q[1], q[0])

    ansatzVec_synth = cudaq.synthesize(ansatzVec, [.59])
    result = cudaq.observe(ansatzVec_synth, hamiltonian)
    print(result.expectation())
    assert np.isclose(result.expectation(), -1.74, atol=1e-2)
