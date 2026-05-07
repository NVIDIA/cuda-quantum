# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
import sys

cudaq.set_target("quake_fake")

qubit_count = 5


@cudaq.kernel
def kernel() -> list[int]:
    qvector = cudaq.qvector(qubit_count)

    for i in range(qubit_count - 1):
        h(qvector[i])

    s(qvector[0])
    r1(math.pi / 2, qvector[1])
    a = mz(qvector)
    return a


@cudaq.kernel
def all_zeros() -> list[int]:
    q = cudaq.qvector(4)
    return mz(q)


@cudaq.kernel
def all_ones() -> list[int]:
    q = cudaq.qvector(4)
    x(q)
    return mz(q)


@cudaq.kernel
def alternating_01() -> list[int]:
    q = cudaq.qvector(4)
    x(q[1])
    x(q[3])
    return mz(q)


@cudaq.kernel
def single_qubit_flip() -> list[int]:
    q = cudaq.qvector(1)
    x(q[0])
    return mz(q)


try:
    res = cudaq.run(kernel)
    assert res is not None
    assert len(res) > 0
    assert len(res[0]) == qubit_count
    for shot in res:
        for val in shot:
            assert val in (0, 1)

    # Deterministic: all qubits stay |0>.
    res = cudaq.run(all_zeros)
    assert len(res) > 0
    for shot in res:
        assert list(shot) == [0, 0, 0,
                              0], f"expected [0,0,0,0], got {list(shot)}"

    # Deterministic: X on all qubits -> all |1>.
    res = cudaq.run(all_ones)
    assert len(res) > 0
    for shot in res:
        assert list(shot) == [1, 1, 1,
                              1], f"expected [1,1,1,1], got {list(shot)}"

    # Deterministic: X on qubits 1 and 3 -> [0,1,0,1].
    res = cudaq.run(alternating_01)
    assert len(res) > 0
    for shot in res:
        assert list(shot) == [0, 1, 0,
                              1], f"expected [0,1,0,1], got {list(shot)}"

    # Deterministic: single qubit X -> [1].
    res = cudaq.run(single_qubit_flip)
    assert len(res) > 0
    for shot in res:
        assert list(shot) == [1], f"expected [1], got {list(shot)}"

except Exception as e:
    print(e)
    sys.exit(1)
