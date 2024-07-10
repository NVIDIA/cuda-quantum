import cudaq
import numpy as np

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

# CUDA-Q supports custom noise models through the definition of
# `KrausChannel`'s. In this case, we will define a set of `KrausOperator`'s
# that  affect the same noise as the `AmplitudeDampingChannel`. This
# channel will model the energy dissipation within our system via
# environmental interactions. With a variable probability, it will
# return the qubit to the |0> state.

# We will begin by defining an empty noise model that we will add
# our Kraus Channel to.
noise = cudaq.NoiseModel()


# We will define our Kraus Operators within functions, as to
# allow for easy control over the noise probability.
def kraus_operators(probability):
    """See Nielsen, Chuang Chapter 8.3.5 for definition source."""
    kraus_0 = np.array([[1, 0], [0, np.sqrt(1 - probability)]],
                       dtype=np.complex128)
    kraus_1 = np.array([[0, np.sqrt(probability)], [0, 0]], dtype=np.complex128)
    return [kraus_0, kraus_1]


# We manually define an amplitude damping channel setting to `1.0`
# the probability of the qubit decaying to the ground state.
amplitude_damping = cudaq.KrausChannel(kraus_operators(1.0))

# We will apply this channel to any Hadamard gate on the qubit.
# In other words, after each Hadamard on the qubit, there will be a
# probability of `1.0` that the qubit decays back to ground.
noise.add_channel('h', [0], amplitude_damping)


@cudaq.kernel
def kernel():
    qubit = cudaq.qubit()
    # Then we apply a Hadamard gate to the qubit.
    # This will bring it to `1/sqrt(2) (|0> + |1>)`, where it will remain
    # with a probability of `1 - p = 0.0`.
    h(qubit)
    # Measure.
    mz(qubit)


# Now we're ready to run the noisy simulation of our kernel.
# Note: We must pass the noise model to sample via keyword.
noisy_result = cudaq.sample(kernel, noise_model=noise)
print(noisy_result)

# Our results should show all measurements in the |0> state, indicating
# that the noise has successfully impacted the system.

# To confirm this, we can run the simulation again without noise.
# The qubit will now have a 50/50 mix of measurements between
# |0> and |1>.
noiseless_result = cudaq.sample(kernel)
print(noiseless_result)
