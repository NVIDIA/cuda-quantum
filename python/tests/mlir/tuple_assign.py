# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_tuple_assign():

    @cudaq.kernel
    def test():
        q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        x(q, s)
        swap.ctrl(q, r, s)

    print(test)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.swap {{\[}}%[[VAL_0]]] %[[VAL_1]], %[[VAL_2]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }
