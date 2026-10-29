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


# [Begin Options Kernel]
# A hyper-edge arises naturally when one fault trips both an
# X-type and a Z-type parity check. This circuit prepares a Bell pair |Phi+>,
# the +1 eigenstate of both XX and ZZ, and measures each stabilizer with its
# own ancilla. A Y error anti-commutes with both checks and flips the data
# readout, lighting up three detectors at once. Because Y = X * Z, that
# hyper-edge decomposes into the separate X and Z edges seeded by the
# accompanying single-qubit errors.
@cudaq.kernel
def correlated_checks():
    data = cudaq.qvector(2)
    z_anc = cudaq.qubit()
    x_anc = cudaq.qubit()

    # Prepare |00> + |11>
    h(data[0])
    x.ctrl(data[0], data[1])

    cudaq.apply_noise(cudaq.XError, 0.01, data[0])
    cudaq.apply_noise(cudaq.ZError, 0.01, data[0])
    cudaq.apply_noise(cudaq.YError, 0.02, data[0])

    # ZZ parity check: the data qubits control the ancilla.
    x.ctrl(data[0], z_anc)
    x.ctrl(data[1], z_anc)
    z_syndrome = mz(z_anc)

    # XX parity check: the ancilla controls the data, read out in the X basis.
    h(x_anc)
    x.ctrl(x_anc, data[0])
    x.ctrl(x_anc, data[1])
    h(x_anc)
    x_syndrome = mz(x_anc)

    final = mz(data)
    cudaq.detector(z_syndrome)
    cudaq.detector(x_syndrome)
    cudaq.detector(final[0], final[1])


# [End Options Kernel]

# [Begin Generate]
# Generate the detector error model. A noise model must be supplied for the
# in-kernel `apply_noise` mechanisms to take effect. The result's `dem` member
# contains Stim `.dem` text; pass it to `stim.DetectorErrorModel(result.dem)`
# to drive a decoder. Converting the result to a string also returns this text.
noise = cudaq.NoiseModel()
result = cudaq.dem_from_kernel(memory_experiment, 2, noise_model=noise)
print(f"Memory experiment DEM:\n{result}")
# [End Generate]

print()
# [Begin Options]
# Pass DEM options as keyword arguments to control the Stim error analyzer.
# decompose_errors=True splits hyper-edge mechanisms (three or more detectors)
# into pairs of graph-like edges, which is required by most MWPM decoders.
dem_raw = cudaq.dem_from_kernel(correlated_checks, noise_model=noise)
dem_decomposed = cudaq.dem_from_kernel(
    correlated_checks,
    noise_model=noise,
    decompose_errors=True,
)
print(f"Raw DEM:\n{dem_raw}")
print(f"Decomposed DEM:\n{dem_decomposed}")
# [End Options]

print()
# [Begin Measurement Matrices]
# `dem_from_kernel` returns a `cudaq.DEMResult`. By default, its `m2d_matrix`
# and `m2o_matrix` properties contain the sparse measurements-to-detectors and
# measurements-to-observables matrices. Both are `scipy.sparse.csr_matrix`
# objects with binary entries, and their columns are indexed by measurement in
# chronological order. Set `return_measurement_matrices=False` to skip their
# computation; both properties will then be `None`.
result = cudaq.dem_from_kernel(
    memory_experiment,
    2,
    noise_model=noise,
)
m2d = result.m2d_matrix
m2o = result.m2o_matrix
# m2d has shape `(num_detectors, num_measurements)`: m2d[d, m] == 1 means
# measurement m contributes to detector d. m2o has shape
# `(num_observables, num_measurements)` with the same convention for observables.
print(f"m2d shape: {m2d.shape}")
print(f"m2o shape: {m2o.shape}")
print(f"m2d:\n{m2d.toarray()}")
print(f"m2o:\n{m2o.toarray()}")
# [End Measurement Matrices]
