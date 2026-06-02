# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Docs]
import cudaq
# [End Docs]


# [Begin Kernel]
# A 3-qubit bit-flip memory experiment. Each round measures the data qubits;
# cross-round detectors pair each measurement with its value in the previous
# round, and a final logical observable reads out the register. In-kernel
# `apply_noise` seeds the error mechanisms the detector error model reports.
@cudaq.kernel
def memory_experiment(rounds: int):
    data = cudaq.qvector(3)
    prev = mz(data)

    for r in range(rounds):
        cudaq.apply_noise(cudaq.XError, 0.01, data[0])
        cudaq.apply_noise(cudaq.XError, 0.01, data[1])
        cudaq.apply_noise(cudaq.XError, 0.01, data[2])

        curr = mz(data)
        # One detector per qubit, pairing this round with the previous one.
        cudaq.detectors(prev, curr)
        prev = curr

    cudaq.logical_observable(prev[0], prev[1], prev[2])


# [End Kernel]

# [Begin Generate]
# Generate the detector error model as Stim `.dem` text. A noise model must be
# supplied for the in-kernel `apply_noise` mechanisms to take effect. Parse the
# text with `stim.DetectorErrorModel(dem)` to drive a decoder.
noise = cudaq.NoiseModel()
dem = cudaq.dem_from_kernel(memory_experiment, 2, noise_model=noise)
print(dem)
# [End Generate]
