# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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
cudaq.set_target("nvidia", option="mqpu")

qubit_count = 5
sample_count = 10000
h = spin.z(0)
parameter_count = qubit_count

# Below we run a circuit for 10000 different input parameters.
parameters = np.random.default_rng(13).uniform(low=0,
                                               high=1,
                                               size=(sample_count,
                                                     parameter_count))

kernel, params = cudaq.make_kernel(list)

qubits = kernel.qalloc(qubit_count)
qubits_list = list(range(qubit_count))

for i in range(qubit_count):
    kernel.rx(params[i], qubits[i])
# [End prepare]

# [Begin single]
import timeit

timeit.timeit(lambda: cudaq.observe(kernel, h, parameters),
              number=1)  # Single GPU result.
# [End single]

# [Begin split]
print('We have', parameters.shape[0],
      'parameters which we would like to execute')

xi = np.split(
    parameters,
    4)  # We split our parameters into 4 arrays since we have 4 GPUs available.

print('We split this into', len(xi), 'batches of', xi[0].shape[0], ',',
      xi[1].shape[0], ',', xi[2].shape[0], ',', xi[3].shape[0])
# [End split]

# [Begin multiple]
# Timing the execution on a single GPU vs 4 GPUs,
# one will see a 4x performance improvement if 4 GPUs are available.

asyncresults = []
num_gpus = cudaq.num_available_gpus()

for i in range(len(xi)):
    for j in range(xi[i].shape[0]):
        qpu_id = i * num_gpus // len(xi)
        asyncresults.append(
            cudaq.observe_async(kernel, h, xi[i][j, :], qpu_id=qpu_id))

result = [res.get() for res in asyncresults]
# [End multiple]
