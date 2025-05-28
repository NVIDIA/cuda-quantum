from cudaq import spin
import cudaq # For observe, kernel, x
import sys # For sys.stderr and sys.exit

# [Begin SpinOp Creation Python]
# The following lines define the spin_op `h` as shown in the documentation.
h_py_example = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
    0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
# [End SpinOp Creation Python]

# [Begin SpinOp Creation Python Execution]
if __name__ == "__main__":
    print(f"Python Spin Operator h_py_example:\n{h_py_example}")

    if h_py_example.get_term_count() == 0:
        print("Error: Python Spin operator h_py_example has no terms.", file=sys.stderr)
        sys.exit(1) # Indicate failure

    # Example of using the spin_op in an observe call with a simple ansatz.
    # The spin_op uses qubits 0 and 1, so we need at least 2 qubits.
    @cudaq.kernel
    def ansatz_py():
        q = cudaq.qvector(2)
        x(q[0]) # A simple operation on one of the qubits.

    energy = cudaq.observe(ansatz_py, h_py_example)
    print(f"Observed energy (Python): {energy.expectation()}")
# [End SpinOp Creation Python Execution]