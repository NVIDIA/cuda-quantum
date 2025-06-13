import cudaq

@cudaq.kernel
def qubit_usage_kernel_py():
    # [Begin PyQubitUsage]
    # Allocate a qubit in the |0> state
    q = cudaq.qubit()
    # Put the qubit in a superposition of |0> and |1>
    h(q)
    print("ID = {}".format(q.id())) # prints 0
    
    r = cudaq.qubit()
    print("ID = {}".format(r.id())) # prints 1
    # qubits go out of scope, implicit deallocation
    # [End PyQubitUsage]
    mz(q) # Add measurements for sample
    mz(r)
    # For the re-allocation demonstration in Python, it's implicit with new variable
    q_realloc = cudaq.qubit()
    print("Reallocated q_realloc ID = {}".format(q_realloc.id())) # prints 0
    mz(q_realloc)


if __name__ == "__main__":
    print("Python Qubit Usage Example:")
    counts = cudaq.sample(qubit_usage_kernel_py)
    counts.dump()