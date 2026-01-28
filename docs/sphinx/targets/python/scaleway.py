# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq

# NOTE: Scaleway credentials must be set before running this program.
# Scaleway costs apply.
cudaq.set_target("scaleway")

# The default device is EMU-CUDAQ-H100, a state vector simulator running on an H100 GPU. Users may choose any of
# the available devices by supplying its name with the `machine` parameter.
# For example,
# ```
# cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100")
# ```
# To ensure we keep the same QPU session between runs, we can also specify the `deduplication_id` parameter.
# For example,
# ```
# cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100", deduplication_id="my_unique_id_1234")
# ```
# Users may also specify QPU session duration limits with the `max_duration` and `max_idle_duration` parameters.
# For example,
# ```
# cudaq.set_target("scaleway", machine="EMU-CUDAQ-H100", max_duration="30m", max_idle_duration="5m")
# ```

# Create the kernel we'd like to execute
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])


# Execute and print out the results.

# Option A:
# By using the asynchronous `cudaq.sample_async`, the remaining
# classical code will be executed while the job is being handled
# by Scaleway.
async_results = cudaq.sample_async(kernel)
# ... more classical code to run ...

async_counts = async_results.get()
print(async_counts)

# Option B:
# By using the synchronous `cudaq.sample`, the execution of
# any remaining classical code in the file will occur only
# after the job has been returned from Scaleway QaaS.
counts = cudaq.sample(kernel)
print(counts)
