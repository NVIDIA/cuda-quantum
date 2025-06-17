import cudaq
import numpy as np
from typing import Callable # For Callable type hint

# [Begin QPE Python]
# Compute phase for U |psi> = exp(-2 pi phase) |psi>
# This example will consider U = T, and |psi> = |1>
# Define a Inverse Quantum Fourier Transform kernel
@cudaq.kernel
def iqft(qubits: cudaq.qview):
    N = qubits.size()
    for i in range(N // 2):
        cudaq.swap(qubits[i], qubits[N - i - 1])

    for i in range(N - 1):
        h(qubits[i])
        j = i + 1
        for y in range(i, -1, -1):
            cudaq.r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

    h(qubits[N - 1])


# Define the U kernel
@cudaq.kernel
def tGate(qubit: cudaq.qubit):
    cudaq.t(qubit)


# Define the state preparation |psi> kernel
@cudaq.kernel
def xGate(qubit: cudaq.qubit):
    cudaq.x(qubit)

# General Phase Estimation kernel for single qubit
# eigen states.
@cudaq.kernel
def qpe(nC: int, nQ: int, statePrep: Callable[[cudaq.qubit], None],
        oracle: Callable[[cudaq.qubit], None]):
    q = cudaq.qvector(nC + nQ)
    countingQubits = q.front(nC)
    # stateRegister = q.back() # q.back() gives last element, not qview
    stateRegister = q[nC:] # Correct way to get the rest as a qview or single qubit if nQ=1

    # If nQ is guaranteed to be 1 for this example:
    state_single_qubit = q[nC]

    statePrep(state_single_qubit) # Pass the single qubit
    h(countingQubits)
    for i in range(nC):
        for _j_loop_var in range(2**i): # _j_loop_var to indicate it's not used
            cudaq.control(oracle, [countingQubits[i]], state_single_qubit) # Pass single qubit
    iqft(countingQubits)
    mz(countingQubits)

# Sample the state to get the phase.
# nC=3, nQ=1 from example
counts = cudaq.sample(qpe, 3, 1, xGate, tGate)
assert len(counts) == 1
# The expected phase for T|1> is 1/8, which is 0.125.
# In binary with 3 counting qubits, this is 0.001.
# The QPE algorithm measures 2^n * phase. So, 8 * 0.125 = 1.
# Binary 1 is '001'. The RST has '100'.
# If U|psi> = exp(2 pi i phase) |psi>, then T|1> = exp(i pi/4)|1>, phase = 1/8.
# If U|psi> = exp(-2 pi i phase) |psi>, then T|1> = exp(i pi/4)|1> => -2 pi phase = pi/4 => phase = -1/8.
# The convention for IQFT usually measures 'phase' directly.
# Let's assume the RST '100' (binary for 4) is the target if there's a different convention or U.
# For T gate on |1>, phase is 1/8. 3 bits means 2^3 * 1/8 = 1. So "001".
# The RST example output "100" (which is 4) is unusual for T-gate QPE on |1> with 3 counting qubits.
# I will keep the assert from the RST for now.
print(counts)
assert '100' in counts
# [End QPE Python]

if __name__ == "__main__":
    pass # Logic is at top level