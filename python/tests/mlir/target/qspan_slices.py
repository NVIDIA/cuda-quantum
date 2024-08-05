# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s | FileCheck %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate | FileCheck %s



import cudaq


@cudaq.kernel
def bar(qubits: cudaq.qview):
    controls = qubits.front(qubits.size() - 1)
    target = qubits.back()
    x.ctrl(controls, target)


@cudaq.kernel
def foo():
    q = cudaq.qvector(4)
    x(q)
    bar(q)
    result = mz(q)


result = cudaq.sample(foo)
print(result.most_probable())
assert '1110' == result.most_probable()

# CHECK: 1110
