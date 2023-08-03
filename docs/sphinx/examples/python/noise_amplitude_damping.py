import cudaq

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA Quantum supports several different models of noise. In this case,
# we will examine the modeling of energy dissipation within our system
# via environmental interactions. The result of this "amplitude damping"
# is to return the qubit to the |0> state with a user-specified probability.

# We will begin by defining an empty noise model that we will add
# these damping channels to.
noise = cudaq.NoiseModel()

# Amplitude damping channel with `0.0` probability of the qubit
# decaying to the ground state.
amplitude_damping_zero = cudaq.AmplitudeDampingChannel(0.0)
# We will apply this first channel to any X-gate on qubit 0. Meaning,
# for each X-gate on the 0-th qubit, the qubit will have a `0.0`
# probability of decaying to the ground state.
noise.add_channel('x', [0], amplitude_damping_zero)

# Amplitude damping channel with `1.0` probability of the qubit
# decaying to the ground state.
amplitude_damping_one = cudaq.AmplitudeDampingChannel(1.0)
# We will apply this channel to any Hadamard gate on qubit 1.
# Meaning, after each Hadamard on the first qubit, there will
# be a probability of `1.0` that the qubit decays back to ground.
noise.add_channel('h', [1], amplitude_damping_one)

# Now we may define our simple kernel function and allocate a register
# of qubits to it.
kernel = cudaq.make_kernel()
qubits = kernel.qalloc(2)

# First we apply an X-gate to qubit 0.
# This will bring the qubit to the |1> state, where it will remain
# with a probability of `1 - p = 1.0`.
kernel.x(qubits[0])

# Now we apply a Hadamard gate to qubit 1.
# This will bring it to `1/sqrt(2) (|0> + |1>)`, where it will remain
# with a probability of `1 - p = 0.0`.
kernel.h(qubits[1])

# Now we're ready to run the noisy simulation of our kernel.
# Note: We must pass the noise model to sample via key-word.
noisy_result = cudaq.sample(kernel, noise_model=noise)
noisy_result.dump()

# Our results should show all measurements in the |10> state, indicating
# that the noise has successfully impacted the system.

# To confirm this, we can run the simulation again without noise.
# The 0-th qubit should still be in the |1> state, but qubit 1 will
# now have a 50/50 mix of measurements between |0> and |1>.
noiseless_result = cudaq.sample(kernel)
noiseless_result.dump()
