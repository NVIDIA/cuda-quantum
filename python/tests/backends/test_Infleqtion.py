# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, pytest, os
from cudaq import spin
import numpy as np

## NOTE: Comment the following line which skips these tests in order to run in
# local dev environment after setting the API key
## NOTE: Superstaq costs apply
pytestmark = pytest.mark.skip("Infleqtion / Superstaq API key required")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    cudaq.set_target("infleqtion")
    yield "Running the tests."
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


def assert_close(got) -> bool:
    return got < -1.5 and got > -1.9


def test_simple_kernel():

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])
        mz(qubits)

    counts = cudaq.sample(bell)
    assert len(counts) == 2
    assert "00" in counts
    assert "11" in counts


def test_all_gates():

    @cudaq.kernel
    def all_gates():
        q = cudaq.qubit()
        h(q)
        x(q)
        y(q)
        z(q)
        r1(np.pi, q)
        rx(np.pi, q)
        ry(np.pi, q)
        rz(np.pi, q)
        s(q)
        t(q)
        u3(0.0, np.pi / 2, np.pi, q)
        mz(q)

        qvec = cudaq.qvector(2)
        x(qvec[0])
        swap(qvec[0], qvec[1])
        mz(qvec)

        ## control modifiers
        qubits = cudaq.qvector(2)
        h.ctrl(qubits[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])
        y.ctrl(qubits[0], qubits[1])
        z.ctrl(qubits[1], qubits[0])
        r1.ctrl(np.pi / 2, qubits[0], qubits[1])
        rx.ctrl(np.pi / 4, qubits[1], qubits[0])
        ry.ctrl(np.pi / 8, qubits[0], qubits[1])
        rz.ctrl(np.pi, qubits[1], qubits[0])
        s.ctrl(qubits[0], qubits[1])
        t.ctrl(qubits[1], qubits[0])
        u3.ctrl(0.0, np.pi / 2, np.pi, qubits[0], qubits[1])
        mz(qubits)

        qreg = cudaq.qvector(3)
        x(qreg[0])
        x(qreg[1])
        swap.ctrl(qreg[0], qreg[1], qreg[2])
        mz(qreg)

        ## adjoint modifiers
        r = cudaq.qubit()
        r1.adj(np.pi, r)
        rx.adj(np.pi / 2, r)
        ry.adj(np.pi / 4, r)
        rz.adj(np.pi / 8, r)
        s.adj(r)
        t.adj(r)
        mz(r)

    # Test here is that this runs
    cudaq.sample(all_gates).dump()


def test_multiple_qvector():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        ancilla = cudaq.qvector(2)
        x(qubits)
        h(ancilla)
        mz(ancilla)

    # Test here is that this runs
    cudaq.sample(kernel).dump()


def test_multiple_measure():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(4)
        a = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        h(a)
        cx(q[1], a[0])
        mz(q[1])
        mz(q[0])
        mz(a)

    # Test here is that this runs
    cudaq.sample(kernel).dump()


def test_observe():
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def ansatz(theta: float):
        qreg = cudaq.qvector(2)
        x(qreg[0])
        ry(theta, qreg[1])
        x.ctrl(qreg[1], qreg[0])

    # Define its spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    res = cudaq.observe(ansatz, hamiltonian, .59, shots_count=2048)
    ## Need to adjust expectation value range
    # assert assert_close(res.expectation())
    print(res.expectation())


def test_exp_pauli():

    @cudaq.kernel
    def test():
        q = cudaq.qvector(2)
        exp_pauli(1.0, q, "XX")
        mz(q)

    counts = cudaq.sample(test)
    counts.dump()
    assert '00' in counts
    assert '11' in counts
    assert not '01' in counts
    assert not '10' in counts


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
