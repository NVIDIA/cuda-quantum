import cudaq


@cudaq.kernel
def spooky_kernel():
    cudaq.qubit()
