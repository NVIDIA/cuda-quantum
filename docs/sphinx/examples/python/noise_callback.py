import cudaq
import numpy as np

# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

noise = cudaq.NoiseModel()

# Noise model callback function
def rx_noise(qubits, params):
    # Model a pulse-length based rotation gate:
    # the bigger the angle, the longer the pulse, i.e., more amplitude damping.
    angle = params[0]
    angle = angle % (2 * np.pi)
    # Damping rate is linearly proportional to the angle
    damping_rate = np.abs(angle / (2 * np.pi))
    print(f"Angle = {angle}, amplitude damping rate = {damping_rate}.")
    return cudaq.AmplitudeDampingChannel(damping_rate)

# Bind the noise model callback function to the 'rx' gate
noise.add_channel('rx', rx_noise)


@cudaq.kernel
def kernel(angle: float):
    qubit = cudaq.qubit()
    rx(angle, qubit)
    mz(qubit)


# Now we're ready to run the noisy simulation of our kernel.
# Note: We must pass the noise model to sample via keyword.
noisy_result = cudaq.sample(kernel, np.pi, noise_model=noise)
print(noisy_result)

# Our results should show measurements in both the |0> and |1> states, indicating
# that the noise has successfully impacted the system. 
# Note: a rx(pi) is equivalent to a Pauli X gate, and thus, it should be 
# in the |1> state if no noise is present.

# To confirm this, we can run the simulation again without noise.
noiseless_result = cudaq.sample(kernel, np.pi)
print(noiseless_result)
