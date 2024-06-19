# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

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
