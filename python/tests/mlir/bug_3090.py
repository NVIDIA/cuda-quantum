# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ python3 %s | FileCheck %s

import cudaq

cudaq.set_target("density-matrix-cpu")
cudaq.set_random_seed(13)


@cudaq.kernel
def foo() -> int:
    q = cudaq.qvector(8)
    cudaq.apply_noise(cudaq.XError, 0.9, q[0])

    x(q)
    retval = 0
    for i in range(8):
        bit = mz(q[i])
        retval = retval * 2
        if bit == 0:
            retval = retval + 0
        else:
            retval = retval + 1
    return retval


results = cudaq.run(foo, shots_count=2)
print("Without noise model:")
print(results)

results = cudaq.run(foo, shots_count=2, noise_model=cudaq.NoiseModel())
print("With noise model:")
print(results)
cudaq.reset_target()

# CHECK: [warning] apply_noise called but no noise model provided.
# CHECK: [warning] apply_noise called but no noise model provided.
# CHECK: Without noise model:
# CHECK: [255, 255]
# CHECK: With noise model:
# CHECK: [127, 127]
