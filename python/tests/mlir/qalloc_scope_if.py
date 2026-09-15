# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_qalloc_freed_at_end_of_if():
    """A qubit allocated in an `if` branch is freed when the branch ends."""

    @cudaq.kernel
    def kernel(b: bool):
        r = cudaq.qubit()
        if b:
            q = cudaq.qvector(2)
            h(q[0])
            x.ctrl(q[0], r)
        mz(r)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel..
# CHECK-SAME:      (%[[VAL_0:.*]]: i1)
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           cc.if(%[[VAL_0]]) {
# CHECK:             cc.scope {
# CHECK:               %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:               quake.dealloc %[[VAL_2]] : !quake.veq<2>
# CHECK:             }
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_1]] : !quake.ref
