import cudaq, numpy as np
from typing import Callable


# Compute phase for U |psi> = exp(-2 pi phase) |psi>
# This example will consider U = T, and |psi> = |1>
# Define a Inverse Quantum Fourier Transform kernel
@cudaq.kernel
def iqft(qubits: cudaq.qview):
    N = qubits.size()
    for i in range(N // 2):
        swap(qubits[i], qubits[N - i - 1])

    for i in range(N - 1):
        h(qubits[i])
        j = i + 1
        for y in range(i, -1, -1):
            r1.ctrl(-np.pi / 2 ** (j - y), qubits[j], qubits[y])

    h(qubits[N - 1])


# Define the U kernel
@cudaq.kernel
def tGate(qubit: cudaq.qubit):
    t(qubit)


# Define the state preparation |psi> kernel
@cudaq.kernel
def xGate(qubit: cudaq.qubit):
    x(qubit)


# General Phase Estimation kernel for single qubit
# eigen states.
@cudaq.kernel
def qpe(
    nC: int,
    nQ: int,
    statePrep: Callable[[cudaq.qubit], None],
    oracle: Callable[[cudaq.qubit], None],
):
    q = cudaq.qvector(nC + nQ)
    countingQubits = q.front(nC)
    stateRegister = q.back()
    statePrep(stateRegister)
    h(countingQubits)
    for i in range(nC):
        for j in range(2**i):
            cudaq.control(oracle, [countingQubits[i]], stateRegister)
    iqft(countingQubits)
    mz(countingQubits)


# Sample the state to get the phase.
counts = cudaq.sample(qpe, 3, 1, xGate, tGate)
assert len(counts) == 1
assert "100" in counts
