# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Dynamic Noise Channel Callback
# This example shows how to create a noise channel that depends on
# the gate operands and parameters.

import cudaq


def noise_callback(qubits, params):
    """
    Dynamic noise channel callback.

    Args:
        qubits: List of qubit indices the gate operates on.
        params: List of gate parameters (e.g., rotation angles).

    Returns:
        A cudaq.kraus_channel to apply after the gate.
    """
    # Example: Stronger noise for larger rotation angles
    angle = params[0] if params else 0.0
    noise_strength = min(abs(angle) / 3.14159 * 0.1, 0.1)

    # Return a depolarization channel with angle-dependent strength
    return cudaq.DepolarizationChannel(noise_strength)


# Add the dynamic noise channel to the 'rx' gate
noise = cudaq.NoiseModel()
noise.add_channel('rx', noise_callback)

# Use the noise model
cudaq.set_noise(noise)

@cudaq.kernel
def kernel(theta: float):
    q = cudaq.qubit()
    rx(theta, q)
    mz(q)

# The noise strength will vary based on the rotation angle
for angle in [0.1, 0.5, 1.0, 2.0, 3.14]:
    cudaq.set_noise(noise)
    counts = cudaq.sample(kernel, angle, shots_count=1000)
    print(f"Angle: {angle:.3f}, Counts: {counts}")

cudaq.unset_noise()