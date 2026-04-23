# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s | FileCheck %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate | FileCheck %s

import cudaq, numpy
from typing import Callable


@cudaq.kernel
def bar(q: cudaq.qubit):
    x(q)


@cudaq.kernel
def foo1(func: Callable[[cudaq.qubit], None], size: int):
    q = cudaq.qvector(size)
    func(q[0])


@cudaq.kernel
def baz(qs: cudaq.qvector, angle: float, adj: list[bool]):
    for idx, is_adj in enumerate(adj):
        if is_adj:
            ry(-angle, qs[idx])
        else:
            ry(angle, qs[idx])


@cudaq.kernel
def foo2(adj: list[bool], set_controls: bool):
    controls, targets = cudaq.qvector(2), cudaq.qvector(len(adj))
    if set_controls:
        x(controls)
    cudaq.control(baz, controls, targets, 1, adj)


@cudaq.kernel
def foo3(adj: list[bool], uncompute: bool):
    targets = cudaq.qvector(len(adj))
    cudaq.adjoint(baz, targets, 1, adj)
    if uncompute:
        baz(targets, 1, adj)


@cudaq.kernel
def foo4(func: Callable[[cudaq.qvector, float, list[bool]], None],
         adj: list[bool], set_controls: bool):
    controls, targets = cudaq.qvector(2), cudaq.qvector(len(adj))
    if set_controls:
        x(controls)
    cudaq.control(func, controls, targets, 1, adj)


@cudaq.kernel
def foo5(func: Callable[[cudaq.qvector, float, list[bool]], None],
         adj: list[bool], uncompute: bool):
    targets = cudaq.qvector(len(adj))
    cudaq.adjoint(func, targets, 1, adj)
    if uncompute:
        func(targets, 1, adj)


out = cudaq.sample(foo1, bar, 1)
assert len(out) == 1 and '1' in out
print("test1 - most probable: " + out.most_probable())

out = cudaq.sample(foo2, [True, False, True], False)
assert len(out) == 1 and '1' not in out.most_probable()
print("test2 - most probable: " + str(out.most_probable()))

out = cudaq.sample(foo2, [True, False, True], True)
assert len(out) == 8
print("test3 - superposition of length " + str(len(out)))

out = cudaq.sample(foo3, [True, False, True], False)
assert len(out) == 8
print("test4 - superposition of length " + str(len(out)))

out = cudaq.sample(foo3, [True, False, True], True)
assert len(out) == 1 and '1' not in out.most_probable()
print("test5 - most probable: " + str(out.most_probable()))

out = cudaq.sample(foo4, baz, [True, False, True], False)
assert len(out) == 1 and '1' not in out.most_probable()
print("test6 - most probable: " + str(out.most_probable()))

out = cudaq.sample(foo4, baz, [True, False, True], True)
assert len(out) == 8
print("test7 - superposition of length " + str(len(out)))

out = cudaq.sample(foo5, baz, [True, False, True], False)
assert len(out) == 8
print("test8 - superposition of length " + str(len(out)))

out = cudaq.sample(foo5, baz, [True, False, True], True)
assert len(out) == 1 and '1' not in out.most_probable()
print("test9 - most probable: " + str(out.most_probable()))

# CHECK: test1 - most probable: 1
# CHECK: test2 - most probable: 00000
# CHECK: test3 - superposition of length 8
# CHECK: test4 - superposition of length 8
# CHECK: test5 - most probable: 000
# CHECK: test6 - most probable: 00000
# CHECK: test7 - superposition of length 8
# CHECK: test8 - superposition of length 8
# CHECK: test9 - most probable: 000

# FIXME: fails per issue https://github.com/NVIDIA/cuda-quantum/issues/3499
# Fix the issue, then uncomment the following code and add llvm-lit checks for tests 10 - 13.
'''
@cudaq.kernel
def controlled(kernel: Callable[[cudaq.qvector, float, list[bool]], None], 
               cs: cudaq.qvector, qs: cudaq.qvector, angle: float, adj: list[bool]):
    cudaq.control(kernel, cs, qs, angle, adj)

@cudaq.kernel
def adjointed(kernel: Callable[[cudaq.qvector, float, list[bool]], None], 
               qs: cudaq.qvector, angle: float, adj: list[bool]):
    cudaq.adjoint(kernel, qs, angle, adj)

@cudaq.kernel
def foo6(func: Callable[[cudaq.qvector, float, list[bool]], None], adj: list[bool], set_controls: bool):
    controls, targets = cudaq.qvector(2), cudaq.qvector(len(adj))
    if set_controls:
        x(controls)
    controlled(func, controls, targets, 1, adj)
    mz(targets)

@cudaq.kernel
def foo7(func: Callable[[cudaq.qvector, float, list[bool]], None], adj: list[bool], uncompute: bool):
    targets = cudaq.qvector(len(adj))
    adjointed(func, targets, 1, adj)
    if uncompute:
        func(targets, 1, adj)
    mz(targets)

out = cudaq.sample(foo8, baz, [True, False, True], False)
assert len(out) == 1 and '000' in out
print("test10 - most probable: " + str(out.most_probable()))

out = cudaq.sample(foo8, baz, [True, False, True], True)
assert len(out) > 1
print("test11 - superposition of length " + str(len(out)))

out = cudaq.sample(foo9, baz, [True, False, True], False)
assert len(out) > 1
print("test12 - superposition of length " + str(len(out)))

out = cudaq.sample(foo9, baz, [True, False, True], True)
assert len(out) == 1 and '000' in out
print("test13 - most probable: " + str(out.most_probable()))
'''
