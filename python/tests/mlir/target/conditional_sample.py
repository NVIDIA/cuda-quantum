# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../.. python3 %s
# NOTE: The machine arg is not reflecting correctly with the following command, gives
#       RuntimeError: `run` is not yet supported on this target.
# SKIPPED: PYTHONPATH=../../.. python3 %s --target quantinuum --quantinuum-machine Helios-1SC --emulate

import cudaq


@cudaq.kernel
def kernel() -> bool:
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

    return mz(q[2])


results = cudaq.run(kernel, shots_count=10)
assert len(results) == 10
assert all(res for res in results)


@cudaq.kernel
def kernel1() -> list[bool]:
    data = cudaq.qvector(2)
    aux = cudaq.qvector(2)
    bits = mz(aux)
    # Nothing occurs since aux == |00>
    if bits[0]:
        x(data[0])

    return mz(data)


results = cudaq.run(kernel1, shots_count=10)
for res in results:
    assert res == [False, False]


@cudaq.kernel
def kernel2() -> list[bool]:
    data = cudaq.qvector(2)
    aux = cudaq.qvector(2)
    bits = mz(aux)
    # Nothing occurs since aux == |00>
    # Write the condition a bit differently
    if bits[0] == True:
        x(data[0])

    return mz(data)


results = cudaq.run(kernel2, shots_count=10)
for res in results:
    assert res == [False, False]


@cudaq.kernel
def kernel3() -> list[bool]:
    data = cudaq.qvector(2)
    aux = cudaq.qvector(2)
    bits = mz(aux)
    # Now, this should have some effects
    if not bits[0]:
        x(data[0])

    return mz(data)


results = cudaq.run(kernel3, shots_count=10)
for res in results:
    assert res == [True, False]


@cudaq.kernel
def kernel4(checkVal: bool) -> list[bool]:
    data = cudaq.qvector(2)
    aux = cudaq.qubit()
    bit = mz(aux)
    if bit != checkVal:
        x(data[0])
    return mz(data)


results = cudaq.run(kernel4, True, shots_count=10)
for res in results:
    assert res == [True, False]

results = cudaq.run(kernel4, False, shots_count=10)
for res in results:
    assert res == [False, False]


@cudaq.kernel
def kernel5(checkVal: int) -> list[bool]:
    # Check bool -> int conversion in == comparison.
    data = cudaq.qvector(2)
    aux = cudaq.qubit()
    bit = mz(aux)
    if bit == checkVal:
        x(data[0])
    return mz(data)


results = cudaq.run(kernel5, 0, shots_count=10)
for res in results:
    assert res == [True, False]

results = cudaq.run(kernel5, 1, shots_count=10)
for res in results:
    assert res == [False, False]


@cudaq.kernel
def kernel6(checkVal: int) -> list[bool]:
    # Check bool -> int conversion in != comparison.
    data = cudaq.qvector(2)
    aux = cudaq.qubit()
    bit = mz(aux)
    if bit != checkVal:
        x(data[0])
    return mz(data)


results = cudaq.run(kernel6, 1, shots_count=10)
for res in results:
    assert res == [True, False]

results = cudaq.run(kernel6, 0, shots_count=10)
for res in results:
    assert res == [False, False]
