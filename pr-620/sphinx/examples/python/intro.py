import cudaq

# We begin by defining the `Kernel` that we will construct our
# program with.
kernel = cudaq.make_kernel()

# Next, we can allocate qubits to the kernel via `qalloc(qubit_count)`.
# An empty call to `qalloc` will return a single qubit.
qubit = kernel.qalloc()

# Now we can begin adding instructions to apply to this qubit!
# Here we'll just add every non-parameterized
# single qubit gate that is supported by CUDA Quantum.
kernel.h(qubit)
kernel.x(qubit)
kernel.y(qubit)
kernel.z(qubit)
kernel.t(qubit)
kernel.s(qubit)

# Next, we add a measurement to the kernel so that we can sample
# the measurement results on our simulator!
kernel.mz(qubit)

# Finally, we can execute this kernel on the state vector simulator
# by calling `cudaq.sample`. This will execute the provided kernel
# `shots_count` number of times and return the sampled distribution
# as a `cudaq.SampleResult` dictionary.
result = cudaq.sample(kernel)

# Now let's take a look at the `SampleResult` we've gotten back!
print(result)  # or result.dump()
