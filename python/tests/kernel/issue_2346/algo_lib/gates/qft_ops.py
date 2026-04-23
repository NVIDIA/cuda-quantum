import cudaq


@cudaq.kernel
def qft_kernel():
    cudaq.qubit()
