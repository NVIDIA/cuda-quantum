import cudaq

@cudaq.kernel
def qvector_usage_kernel_py():
    # [Begin PyQvectorUsage]
    # Allocate 20 qubits, vector-like semantics
    q = cudaq.qvector(20)
    # Get the first qubit 
    first = q.front()
    h(first) # Example operation

    # Get the first 5 qubits 
    first_5 = q.front(5)
    for qubit_in_view in first_5:
      h(qubit_in_view)
    
    # Get the last qubit 
    last = q.back()
    x(last) # Example operation

    # Can loop over qubits with size or len function 
    for i in range(len(q)):
      h(q[i]) # .. do something with q[i] ..
    # Range based for loop 
    for qb in q:
      x(qb) # .. do something with qb .. 
    # [End PyQvectorUsage]
    mz(q) # Measure all for sampling

if __name__ == "__main__":
    print("Python Qvector Usage Example:")
    counts = cudaq.sample(qvector_usage_kernel_py)
    counts.dump()