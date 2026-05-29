# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin prepare]
import cudaq
from cudaq import spin
import numpy as np

if cudaq.num_available_gpus() == 0:
    print("This example requires a GPU to run. No GPU detected.")
    exit(0)

np.random.seed(1)
cudaq.set_target("nvidia")

qubit_count = 5
sample_count = 10000
h = spin.z(0)
parameter_count = qubit_count

# prepare 10000 different input parameter sets.
parameters = np.random.default_rng(13).uniform(low=0,
                                               high=1,
                                               size=(sample_count,
                                                     parameter_count))


@cudaq.kernel
def kernel(params: list[float]):

    qubits = cudaq.qvector(5)

    for i in range(5):
        rx(params[i], qubits[i])


# [End prepare]

# [Begin single]
import time

start_time = time.time()
cudaq.observe(kernel, h, parameters)
end_time = time.time()
print(end_time - start_time)

# [End single]

# [Begin split]
print('There are', parameters.shape[0], 'parameter sets to execute')

xi = np.split(
    parameters,
    4)  # Split the parameters into 4 arrays since 4 GPUs are available.

print('Split parameters into', len(xi), 'batches of', xi[0].shape[0], ',',
      xi[1].shape[0], ',', xi[2].shape[0], ',', xi[3].shape[0])
# [End split]

# [Begin multiple]
# Timing the execution on a single GPU vs 4 GPUs,
# one will see a nearly 4x performance improvement if 4 GPUs are available.

cudaq.set_target("nvidia", option="mqpu")
asyncresults = []
num_gpus = cudaq.num_available_gpus()

start_time = time.time()
for i in range(len(xi)):
    for j in range(xi[i].shape[0]):
        qpu_id = i * num_gpus // len(xi)
        asyncresults.append(
            cudaq.observe_async(kernel, h, xi[i][j, :], qpu_id=qpu_id))
result = [res.get() for res in asyncresults]
end_time = time.time()
print(end_time - start_time)
# [End multiple]
