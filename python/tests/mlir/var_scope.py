# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_var_scope():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        var_int = 42
        var_bool = True

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant true
# CHECK:           %[[VAL_1:.*]] = arith.constant 42 : i64
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_3:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i64>
# CHECK:           %[[VAL_4:.*]] = cc.alloca i1
# CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i1>
# CHECK:           return
# CHECK:         }
