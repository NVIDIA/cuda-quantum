# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq


# [Begin Sample_Works]
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@cudaq.kernel
def reset_pattern():
    q = cudaq.qubit()
    h(q)
    mz(q)
    reset(q)
    x(q)


print("Implicit measurements:")
cudaq.sample(bell).dump()

print("\nMid-circuit measurement with reset:")
cudaq.sample(reset_pattern).dump()

print("\nWith explicit_measurements option:")
cudaq.sample(reset_pattern, explicit_measurements=True).dump()
# [End Sample_Works]
''' [Begin SampleWorksOutput]
Implicit measurements:
{ 00:~500 11:~500 }

Mid-circuit measurement with reset:
{ 1:1000 }

With explicit_measurements option:
{ 01:~500 11:~500 }
[End SampleWorksOutput] '''


# [Begin Example1]
@cudaq.kernel
def simple_conditional() -> bool:
    q = cudaq.qvector(2)
    h(q[0])
    r = mz(q[0])
    if r:
        x(q[1])
    return mz(q[1])


results = cudaq.run(simple_conditional, shots_count=100)
n_ones = sum(results)
print(f"Measured |1> {n_ones} out of {len(results)} shots")
# [End Example1]


# [Begin Example2]
@cudaq.kernel
def multi_measure() -> list[bool]:
    q = cudaq.qvector(3)
    h(q)
    r0 = mz(q[0])
    r1 = mz(q[1])
    if r0 and r1:
        x(q[2])
    r2 = mz(q[2])
    return [r0, r1, r2]


results = cudaq.run(multi_measure, shots_count=100)
for shot in results[:5]:
    print(''.join('1' if b else '0' for b in shot))
# [End Example2]


# [Begin Example3]
@cudaq.kernel
def teleport() -> list[bool]:
    results = [False, False, False]
    q = cudaq.qvector(3)
    x(q[0])

    h(q[1])
    x.ctrl(q[1], q[2])

    x.ctrl(q[0], q[1])
    h(q[0])

    results[0] = mz(q[0])
    results[1] = mz(q[1])

    if results[1]:
        x(q[2])
    if results[0]:
        z(q[2])

    results[2] = mz(q[2])
    return results


runs = cudaq.run(teleport, shots_count=100)
assert all(r[2] for r in runs), "Teleportation failed"
print(f"Teleportation succeeded on all {len(runs)} shots")
# [End Example3]

# [Begin Result_Processing]
from collections import Counter

results = cudaq.run(multi_measure, shots_count=1000)
counts = Counter(
    ''.join('1' if bit else '0' for bit in result) for result in results)
print(dict(counts))
# [End Result_Processing]
