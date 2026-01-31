import cudaq

# Set the target at the beginning of the program.
cudaq.set_target("qibo")


# Create the kernel.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector)


# Execute on synchronously on the Qibo cloud and print out the results.
counts = cudaq.sample(kernel)
print(counts)
