# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Docs]
import time
import cudaq
# Use the `nvidia` target
cudaq.set_target("nvidia")

# Let's define a simple kernel that we will add noise to.
qubit_count = 10


@cudaq.kernel
def kernel(qubit_count: int):
    qvector = cudaq.qvector(qubit_count)
    x(qvector)
    mz(qvector)


# Add a simple bit-flip noise channel to X gate
error_probability = 0.01
bit_flip = cudaq.BitFlipChannel(error_probability)

# Add noise channels to our noise model.
noise_model = cudaq.NoiseModel()
# Apply the bit-flip channel to any X-gate on any qubits
noise_model.add_all_qubit_channel("x", bit_flip)

ideal_counts = cudaq.sample(kernel, qubit_count, shots_count=1000)

start = time.time()
# Due to the impact of noise, our measurements will no longer be uniformly
# in the |1...1> state.
noisy_counts = cudaq.sample(kernel,
                            qubit_count,
                            noise_model=noise_model,
                            shots_count=1000)
end = time.time()
noisy_counts.dump()
print(f"Simulation elapsed time: {(end - start) * 1000} ms")