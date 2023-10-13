import cudaq

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA Quantum supports several different models of noise. In this case,
# we will examine the modeling of energy dissipation within our system
# via environmental interactions. The result of this "amplitude damping"
# is to return the qubit to the |0> state with a user-specified probability.

# We will begin by defining an empty noise model that we will add
# our damping channel to.
noise = cudaq.NoiseModel()

# Amplitude damping channel with `1.0` probability of the qubit
# decaying to the ground state.
amplitude_damping = cudaq.AmplitudeDampingChannel(1.0)
# We will apply this channel to any Hadamard gate on the qubit.
# Meaning, after each Hadamard on the qubit, there will be a
# probability of `1.0` that the qubit decays back to ground.
noise.add_channel('h', [0], amplitude_damping)

# Now we may define our simple kernel function and allocate a qubit.
kernel = cudaq.make_kernel()
qubit = kernel.qalloc()

# Then we apply a Hadamard gate to the qubit.
# This will bring it to `1/sqrt(2) (|0> + |1>)`, where it will remain
# with a probability of `1 - p = 0.0`.
kernel.h(qubit)

# Measure.
kernel.mz(qubit)

# Now we're ready to run the noisy simulation of our kernel.
# Note: We must pass the noise model to sample via key-word.
noisy_result = cudaq.sample(kernel, noise_model=noise)
noisy_result.dump()

# Our results should show all measurements in the |0> state, indicating
# that the noise has successfully impacted the system.

# To confirm this, we can run the simulation again without noise.
# The qubit will now have a 50/50 mix of measurements between
# |0> and |1>.
noiseless_result = cudaq.sample(kernel)
noiseless_result.dump()
