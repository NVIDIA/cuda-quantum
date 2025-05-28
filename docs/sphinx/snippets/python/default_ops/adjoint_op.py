import cudaq


def main():
    # Create a kernel and allocate a qubit in a |0> state.
    qubit = cudaq.qubit()

    # Apply the unitary transformation defined by the matrix
    # T = | 1      0     |
    #     | 0  exp(iπ/4) |
    # to the state of the qubit `q`:
    t(qubit)

    # Apply its adjoint transformation defined by the matrix
    # T† = | 1      0     |
    #      | 0  exp(-iπ/4) |
    t.adj(qubit)
    # `qubit` is now again in the initial state |0>.

if __name__ == "__main__":
    main()