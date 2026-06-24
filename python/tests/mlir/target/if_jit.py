# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s | FileCheck %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate | FileCheck %s

import cudaq


@cudaq.kernel
def foo(value: bool):
    q = cudaq.qubit()
    if value:
        x(q)

    result = mz(q)


result = cudaq.sample(foo, True, shots_count=100)
assert '1' == result.most_probable()
print(result.most_probable())

# CHECK: 1
