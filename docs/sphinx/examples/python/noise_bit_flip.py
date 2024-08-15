import cudaq
import numpy as np
# Set the target to our density matrix simulator.
cudaq.set_target('density-matrix-cpu')


noise = cudaq.NoiseModel()
angle_threshold = np.pi/8

def rx_noise(qubits, params):
    print("Applying RX noise to qubits ", qubits, "; angle =", params)
    return cudaq.BitFlipChannel(1.0)

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

# # Our results should show all measurements in the |0> state, indicating
# # that the noise has successfully impacted the system.

# # To confirm this, we can run the simulation again without noise.
# # We should now see the qubit in the |1> state.
# noiseless_result = cudaq.sample(kernel)
# print(noiseless_result)
