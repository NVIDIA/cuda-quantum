import cudaq

# [Begin Intrinsic Modifiers Python]
@cudaq.kernel
def kernel_intrinsic_modifiers_py():
    q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit() # Allocate 3 qubits

    # Apply T operation
    t(q)

    # Apply Tdg operation
    t.adj(q)

    # Apply control Hadamard operation
    # q and r are controls, s is target
    h.ctrl(q,r,s)

    # Error, ctrl requires > 1 qubit operands (target + at least one control)
    # h.ctrl(r) # This line is commented out in the original, keep it so.

    # Add measurements
    mz(q)
    mz(r)
    mz(s)
    print("Python: Intrinsic modifiers kernel executed.")
# [End Intrinsic Modifiers Python]

# [Begin Intrinsic Modifiers Python Execution]
if __name__ == "__main__":
    cudaq.sample(kernel_intrinsic_modifiers_py)
# [End Intrinsic Modifiers Python Execution]