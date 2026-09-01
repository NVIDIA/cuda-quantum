# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import sys

cudaq.set_target("quake_fake")


@cudaq.kernel
def synthesized_early_return(condition: bool) -> bool:
    q = cudaq.qubit()
    h(q)
    if condition:
        return True
    x(q)
    return False


@cudaq.kernel
def synthesized_break(condition: bool) -> bool:
    q = cudaq.qubit()
    for _ in range(2):
        h(q)
        if condition:
            break
        x(q)
    return False


@cudaq.kernel
def synthesized_continue(condition: bool) -> bool:
    q = cudaq.qubit()
    for _ in range(2):
        h(q)
        if condition:
            continue
        x(q)
    return False


@cudaq.kernel
def measurement_early_return() -> bool:
    q = cudaq.qubit()
    x(q)
    if mz(q):
        return True
    h(q)
    return False


@cudaq.kernel
def measurement_break() -> bool:
    q = cudaq.qubit()
    for _ in range(2):
        x(q)
        if mz(q):
            break
        h(q)
    return False


@cudaq.kernel
def measurement_continue() -> bool:
    q = cudaq.qubit()
    for _ in range(2):
        x(q)
        if mz(q):
            continue
        h(q)
    return False


if len(sys.argv) != 2:
    raise ValueError("expected one unwind case")

if sys.argv[1] == "early-return":
    cudaq.run(synthesized_early_return, True, shots_count=1)
elif sys.argv[1] == "break":
    cudaq.run(synthesized_break, True, shots_count=1)
elif sys.argv[1] == "continue":
    cudaq.run(synthesized_continue, True, shots_count=1)
elif sys.argv[1] == "measurement-return":
    cudaq.run(measurement_early_return, shots_count=1)
elif sys.argv[1] == "measurement-break":
    cudaq.run(measurement_break, shots_count=1)
elif sys.argv[1] == "measurement-continue":
    cudaq.run(measurement_continue, shots_count=1)
else:
    raise ValueError(f"unknown unwind case: {sys.argv[1]}")
