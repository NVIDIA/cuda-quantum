import cudaq

# FIXME

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA Quantum supports several different models of noise. In this case,
# we will examine the modeling of decoherence of the qubit state. This
# will occur from "bit flip" errors, wherein the qubit has a user-specified
# probability of undergoing an X-180 rotation.

# We will begin by defining an empty noise model that we will add
# these decoherence channels to.
noise = cudaq.NoiseModel()

# Bit flip channel with `0.0` probability of the qubit flipping 180 degrees.
bit_flip_zero = cudaq.BitFlipChannel(0.0)
# We will apply this first channel to any X-gate on qubit 0. Meaning,
# for each X-gate on the 0-th qubit, the qubit will have a `0.0`
# probability of undergoing an extra X-180 rotation.
noise.add_channel('x', [0], bit_flip_zero)

# Bit flip channel with `1.0` probability of the qubit flipping 180 degrees.
bit_flip_one = cudaq.BitFlipChannel(1.0)
# We will apply this channel to any X gate on qubit 1, with a probability
# of `1.0`.
noise.add_channel('x', [1], bit_flip_one)

# Now we may define our simple kernel function and allocate a register
# of qubits to it.
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(2)

# First we apply an X-gate to qubit 0.
# This will bring the qubit to the |1> state, where it will remain
# with a probability of `1 - p = 1.0`.
kernel.x(qubits[0])

# Now we apply an X-gate to qubit 1.
# It will remain in the |1> state with a probability of `1 - p = 0.0`.
kernel.x(qubits[1])

# Now we're ready to run the noisy simulation of our kernel.
# Note: We must pass the noise model to sample via key-word.
noisy_result = cudaq.sample(kernel, noise_model=noise)
noisy_result.dump()

# Our results should show all measurements in the |10> state, indicating
# that the noise has successfully impacted the system.

# To confirm this, we can run the simulation again without noise.
# We should now see both qubits in the |1> state.
noiseless_result = cudaq.sample(kernel)
noiseless_result.dump()
