import cudaq
import numpy as np
# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')

noise = cudaq.NoiseModel()

def rx_noise(qubits, params):
    # Model a pulse-length based rotation gate:
    # the bigger the angle, the longer the pulse, i.e., more amplitude damping.
    angle = params[0]
    angle = angle % (2 * np.pi)
    # A toy model for demonstration!!
    damping_rate = np.abs(angle / (2 * np.pi))
    print(f"Angle = {angle}, amplitude damping rate = {damping_rate}.")
    return cudaq.AmplitudeDampingChannel(damping_rate)

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

print(cudaq.NoiseModel.add_channel.__doc__)