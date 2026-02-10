# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ python3 %s --cudaq-full-stack-trace 2> %t; cat %t | FileCheck %s -check-prefix=FAIL

import cudaq


@cudaq.kernel
def simple(numQubits: int) -> int:
    qubits = cudaq.qvector(numQubits)
    return 1


cudaq.run(simple, [2])

# FAIL: Invalid runtime argument type.
