import cudaq

# We begin by defining the `Kernel`to build our quantum program
kernel = cudaq.make_kernel()

# Next, we allocate qubits to the kernel via `qalloc(qubit_count)`.
# In this example, we will allocate one qubit.
# An empty call to `qalloc` will return a single qubit.
qubit = kernel.qalloc()

# Now we begin adding instructions to apply to this qubit!
# Here, we'll just add a few of the non-parameterized
# single qubit gates that are supported by CUDA Quantum.
# In addition to the gates below, we could have also added
# the gates representing the adjoint of these operators
# (for example, `kernel.tdg(qubit)`). 
kernel.h(qubit)
kernel.x(qubit)
kernel.y(qubit)
kernel.z(qubit)
kernel.t(qubit)
kernel.s(qubit)

# Next, we add a measurement to the kernel so that we can sample
# the measurement results on our simulator.
kernel.mz(qubit)

# Finally, we execute this kernel on the state vector simulator
# by calling `cudaq.sample`. This will execute the provided kernel
# `shots_count` number of times and return the sampled distribution
# as a `cudaq.SampleResult` dictionary.
result = cudaq.sample(kernel)

# Now let's take a look at the `SampleResult` that we've gotten back!
print(result)  # or result.dump()
