# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq
from qrisp import QuantumFloat, h, measure
from qrisp_cudaq import cudaq_kernel


@cudaq_kernel
def main(k: int):
    a = QuantumFloat(3)
    h(a)  # a is in superposition of 0, 1, ..., 7
    b = QuantumFloat(4)
    b[:] = k
    b += a  # adds all eight values of a to k at once
    return measure(a), measure(b)


results = cudaq.run(main, 5, shots_count=5)
print(results)
# [(1.0, 6.0), (3.0, 8.0), (0.0, 5.0), (4.0, 9.0), (1.0, 6.0)]
# [End Documentation]

# The nightly CUDA-Q integration build only checks that this script exits
# without an error, so also check the results.
assert len(results) == 5
assert all(b == a + 5 for a, b in results)
