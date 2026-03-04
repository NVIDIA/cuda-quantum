# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Docs]
import cudaq
from cudaq import spin

# Use the `nvidia` target
# Other targets capable of trajectory simulation are:
# - `tensornet`
# - `tensornet-mps`
cudaq.set_target("nvidia")


@cudaq.kernel
def kernel():
    q = cudaq.qubit()
    x(q)


# Add a simple bit-flip noise channel to X gate
error_probability = 0.1
bit_flip = cudaq.BitFlipChannel(error_probability)

# Add noise channels to our noise model.
noise_model = cudaq.NoiseModel()
# Apply the bit-flip channel to any X-gate on any qubits
noise_model.add_all_qubit_channel("x", bit_flip)

noisy_exp_val = cudaq.observe(kernel,
                              spin.z(0),
                              noise_model=noise_model,
                              num_trajectories=1024).expectation()
# True expectation: 0.1 - 0.9 = -0.8 (|1> has <Z> of -1 and |1> has <Z> of +1)
print("Noisy <Z> with 1024 trajectories =", noisy_exp_val)

# Rerun with a higher number of trajectories
noisy_exp_val = cudaq.observe(kernel,
                              spin.z(0),
                              noise_model=noise_model,
                              num_trajectories=8192).expectation()
print("Noisy <Z> with 8192 trajectories =", noisy_exp_val)
