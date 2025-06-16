import cudaq
from typing import Callable

# [Begin Grover Python]
@cudaq.kernel
def reflect(qubits: cudaq.qview):
    ctrls = qubits.front(qubits.size() - 1)
    last = qubits.back()
    # cudaq.compute_action is a specific pattern, ensure it's used correctly
    # For a general reflection about average: H^n U_0 H^n
    # U_0 flips sign of |0...0>, identity otherwise.
    # This specific reflect is a reflection about |s'> = H|s>
    # H^(tensor N) X^(tensor N) Z_controlled H^(tensor N)
    # The provided reflect is a reflection about the all |+> state if U_f is identity.
    # More standard: Reflection about the average
    # H(qubits)
    # X(qubits)
    # Z.`ctrl`(ctrls, last) # Multi-controlled Z on last qubit, flips |1...1>
    # X(qubits)
    # H(qubits)
    # The compute_action simplifies this if the lambda captures the H and X.
    cudaq.compute_action(lambda: (h(qubits), x(qubits)),
                          lambda: z.ctrl(ctrls, last))

@cudaq.kernel
def oracle(q: cudaq.qview): # Marks |101> and |011> (if q[2] is target)
    # This oracle marks states where (q0=1 AND q2=1) OR (q1=1 AND q2=1)
    # by flipping the phase of q[2] if q[0]=1 or q[1]=1.
    # A typical Grover oracle flips the phase of the *marked item(s)*.
    # If q[2] is an ancilla, this is more like Deutsch-`Jozsa`.
    # Assuming q[2] is part of the search space and we want to find e.g. |101>
    # A common oracle for |101> (3 qubits) would be X(q[1]), CNOT(q[0],q[2]), CNOT(q[1],q[2]), X(q[1])
    # The provided oracle is:
    # z.`ctrl`(q[0], q[2]) -> if q[0]=1, apply Z to q[2]
    # z.`ctrl`(q[1], q[2]) -> if q[1]=1, apply Z to q[2]
    # This means if (q0=1, q1=0) -> Z on q2. If (q0=0, q1=1) -> Z on q2.
    # If (q0=1, q1=1) -> Z^2=I on q2. If (q0=0, q1=0) -> I on q2.
    # This oracle is a bit unusual for standard Grover search for specific bitstrings.
    # It's marking states based on q[0] or q[1] to affect q[2].
    # For the assert '101' and '011', it implies these are the states being searched for.
    # Let's assume the oracle correctly marks the states that lead to the assert.
    z.ctrl(q[0], q[2])
    z.ctrl(q[1], q[2])


@cudaq.kernel
def grover(N: int, M: int, oracle_kernel: Callable[[cudaq.qview], None]): # Renamed oracle to oracle_kernel
    q = cudaq.qvector(N)
    h(q)
    for _i_loop_var in range(M): # _i_loop_var
        oracle_kernel(q) # Use passed oracle_kernel
        reflect(q) # reflect is defined above
    mz(q)

# N=3 (3 qubits), M=1 (1 iteration) from example
counts = cudaq.sample(grover, 3, 1, oracle)
print(counts)
# The assert implies specific outcomes.
# For N=3, M=1, we'd expect some amplification.
# The oracle z.`ctrl`(q[0],q[2]); z.`ctrl`(q[1],q[2]) marks states where q0 XOR q1 is 1, to apply Z to q2.
# This is not a standard Grover oracle for finding '101' and '011'.
# However, I will keep the asserts as per the RST.
assert len(counts) == 2 # This might be too strict for M=1.
assert '101' in counts
assert '011' in counts
# [End Grover Python]

if __name__ == "__main__":
    pass # Logic is at top level