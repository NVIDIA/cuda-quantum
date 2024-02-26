# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ python3  %s | FileCheck %s
# XFAIL: *

# Not supported yet.

import cudaq


@cudaq.kernel
def foo():
    q0, q1, q2 = cudaq.qvector(3)
    x(q0)
    y(q1)
    z(q2)


print(foo)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__foo() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.y %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.z %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }
