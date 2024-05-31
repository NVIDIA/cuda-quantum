# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s
# RUN: PYTHONPATH=../../.. python3 %s --target quantinuum --emulate

import cudaq


@cudaq.kernel
def kernel():
    q = cudaq.qvector(3)
    x(q[0])
    h(q[1])
    x.ctrl(q[1], q[2])
    x.ctrl(q[0], q[1])
    h(q[0])
    b0 = mz(q[0])
    b1 = mz(q[1])
    if b1:
        x(q[2])
    if b0:
        z(q[2])

    mz(q[2])


counts = cudaq.sample(kernel, shots_count=100)
counts.dump()
resultsOnZero = counts.get_marginal_counts([0])
resultsOnZero.dump()

nOnes = resultsOnZero.count('1')
assert nOnes == 100


@cudaq.kernel
def kernel1():
    data = cudaq.qvector(2)
    aux = cudaq.qvector(2)
    bits = mz(aux)
    # Nothing occurs since aux == |00>
    if bits[0]:
        x(data[0])

    mz(data)


counts = cudaq.sample(kernel1, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["00"] == 100


@cudaq.kernel
def kernel2():
    data = cudaq.qvector(2)
    aux = cudaq.qvector(2)
    bits = mz(aux)
    # Nothing occurs since aux == |00>
    # Write the condition a bit differently
    if bits[0] == True:
        x(data[0])

    mz(data)


counts = cudaq.sample(kernel2, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["00"] == 100


@cudaq.kernel
def kernel3():
    data = cudaq.qvector(2)
    aux = cudaq.qvector(2)
    bits = mz(aux)
    # Now, this should has some effects
    if not bits[0]:
        x(data[0])

    mz(data)


counts = cudaq.sample(kernel3, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["10"] == 100


@cudaq.kernel
def kernel4(checkVal: bool):
    data = cudaq.qvector(2)
    aux = cudaq.qubit()
    bit = mz(aux)
    if bit != checkVal:
        x(data[0])
    mz(data)


counts = cudaq.sample(kernel4, True, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["10"] == 100
counts = cudaq.sample(kernel4, False, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["00"] == 100


@cudaq.kernel
def kernel5(checkVal: int):
    # Check bool -> int conversion in == comparison.
    data = cudaq.qvector(2)
    aux = cudaq.qubit()
    bit = mz(aux)
    if bit == checkVal:
        x(data[0])
    mz(data)


counts = cudaq.sample(kernel5, 0, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["10"] == 100
counts = cudaq.sample(kernel5, 1, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["00"] == 100


@cudaq.kernel
def kernel6(checkVal: int):
    # Check bool -> int conversion in != comparison.
    data = cudaq.qvector(2)
    aux = cudaq.qubit()
    bit = mz(aux)
    if bit != checkVal:
        x(data[0])
    mz(data)


counts = cudaq.sample(kernel6, 1, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["10"] == 100
counts = cudaq.sample(kernel6, 0, shots_count=100)
counts.dump()
assert counts.get_register_counts("__global__")["00"] == 100
