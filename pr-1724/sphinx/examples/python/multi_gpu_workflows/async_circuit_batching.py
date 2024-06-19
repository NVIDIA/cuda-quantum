# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin
import numpy as np

np.random.seed(1)

cudaq.set_target("nvidia-mqpu")

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
