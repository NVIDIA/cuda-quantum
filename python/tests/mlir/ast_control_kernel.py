# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_control_kernel():

    @cudaq.kernel
    def applyX(q: cudaq.qubit):
        x(q)

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        cudaq.control(applyX, [q[0]], q[1])

    print(bell)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__applyX(
# CHECK-SAME:                                        %[[VAL_0:.*]]: !quake.ref) {
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:     func.func @__nvqpp__mlirgen__bell() attributes {"cudaq-entrypoint"} {
# CHECK:      %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.apply @__nvqpp__mlirgen__applyX {{\[}}%[[VAL_1]]] %[[VAL_2]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
