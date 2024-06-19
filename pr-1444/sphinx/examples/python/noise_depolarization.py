import cudaq

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA-Q supports several different models of noise. In this
# case, we will examine the modeling of depolarization noise. This
# depolarization will result in the qubit state decaying into a mix
# of the basis states, |0> and |1>, with a user provided probability.

# We will begin by defining an empty noise model that we will add
# our depolarization channel to.
noise = cudaq.NoiseModel()

# We define a depolarization channel setting the probability
# of the qubit state being scrambled to `1.0`.
depolarization = cudaq.DepolarizationChannel(1.0)

# We will apply the channel to any Y-gate on qubit 0. In other words,
# for each Y-gate on our qubit, the qubit will have a `1.0`
# probability of decaying into a mixed state.
noise.add_channel('y', [0], depolarization)


# Now we define our simple kernel function and allocate
# a qubit to it.
@cudaq.kernel
def kernel():
    qubit = cudaq.qubit()
    # First we apply a Y-gate to the qubit.
    # This will bring it to the |1> state, where it will remain
    # with a probability of `1 - p = 0.0`.
    y(qubit)
    mz(qubit)


# Without noise, the qubit should still be in the |1> state.
counts = cudaq.sample(kernel)
counts.dump()

# With noise, the measurements should be a roughly 50/50
# mix between the |0> and |1> states.
noisy_counts = cudaq.sample(kernel, noise_model=noise)
noisy_counts.dump()
