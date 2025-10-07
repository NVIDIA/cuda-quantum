# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate

# Perform a single test with --target=<target>
# RUN: PYTHONPATH=../../.. python3 %s --target=quantinuum --emulate

import cudaq, numpy
from typing import Callable


@cudaq.kernel
def bar(q: cudaq.qubit, dummy: int):
    ry(numpy.pi, q)

@cudaq.kernel
def baz(q: cudaq.qubit, dummy: int):
    x(q)


@cudaq.kernel
def foo1(func: Callable[[cudaq.qubit, int], None], size: int):
    q = cudaq.qvector(size)
    func(q[0], 5)
    mz(q[0])

@cudaq.kernel
def foo2(func: Callable[[cudaq.qubit, int], None], size: int):
    q = cudaq.qvector(size)
    x(q[1:])
    cudaq.control(func, q[1:], q[0], 5.0)
    mz(q[0])


out = cudaq.sample(foo1, baz, 1)
assert len(out) == 1 and '1' in out
print("test1 most probable: " + str(out.most_probable()))

out = cudaq.sample(foo1, bar, 1)
assert len(out) == 1 and '1' in out
print("test2 most probable: " + str(out.most_probable()))

out = cudaq.sample(foo2, bar, 3)
assert len(out) == 1 and '1' in out
print("test3 most probable: " + str(out.most_probable()))

# CHECK: test1 most probable: 1
# CHECK: test2 most probable: 1
# CHECK: test3 most probable: 1
