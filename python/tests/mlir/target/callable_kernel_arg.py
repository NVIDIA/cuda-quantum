# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s | FileCheck %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate | FileCheck %s

# Perform a single test with --target=<target>
# RUN: PYTHONPATH=../../.. python3 %s --target=quantinuum --emulate | FileCheck %s

import cudaq
from typing import Callable


@cudaq.kernel
def bar(q: cudaq.qubit):
    x(q)


@cudaq.kernel
def baz(q: cudaq.qubit):
    x(q)


@cudaq.kernel
def foo(func: Callable[[cudaq.qubit], None], size: int):
    q = cudaq.qvector(size)
    func(q[0])
    result = mz(q[0])


result = cudaq.sample(foo, baz, 1)
print(result.most_probable())

# CHECK: 1
