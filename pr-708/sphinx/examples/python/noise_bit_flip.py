import cudaq

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA Quantum supports several different models of noise. In this case,
# we will examine the modeling of decoherence of the qubit state. This
# will occur from "bit flip" errors, wherein the qubit has a user-specified
# probability of undergoing an X-180 rotation.

# We will begin by defining an empty noise model that we will add
# these decoherence channels to.
noise = cudaq.NoiseModel()

# Bit flip channel with `1.0` probability of the qubit flipping 180 degrees.
bit_flip = cudaq.BitFlipChannel(1.0)
# We will apply this channel to any X gate on the qubit, giving each X-gate
# a probability of `1.0` of undergoing an extra X-gate.
noise.add_channel('x', [0], bit_flip)

# Now we may define our simple kernel function and allocate a register
# of qubits to it.
kernel = cudaq.make_kernel()
qubit = kernel.qalloc()

# Apply an X-gate to the qubit.
# It will remain in the |1> state with a probability of `1 - p = 0.0`.
kernel.x(qubit)
# Measure.
kernel.mz(qubit)

# Now we're ready to run the noisy simulation of our kernel.
# Note: We must pass the noise model to sample via key-word.
noisy_result = cudaq.sample(kernel, noise_model=noise)
noisy_result.dump()

# Our results should show all measurements in the |0> state, indicating
# that the noise has successfully impacted the system.

# To confirm this, we can run the simulation again without noise.
# We should now see the qubit in the |1> state.
noiseless_result = cudaq.sample(kernel)
noiseless_result.dump()
