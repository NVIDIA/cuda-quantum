# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_tuple_assign():

    @cudaq.kernel
    def fancyCnot(a: cudaq.qubit, b: cudaq.qubit):
        x.ctrl(a, b)

    @cudaq.kernel
    def toffoli():
        q = cudaq.qvector(3)
        ctrl = q.front()
        x(ctrl, q[2])
        cudaq.control(fancyCnot, [ctrl], q[1], q[2])

    print(toffoli)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__toffoli..
# CHECK-SAME:      (%[[ARG0:.*]]: !cc.callable<(!quake.ref, !quake.ref) -> ()> {quake.pylifted}) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]][0] : (!quake.veq<3>) -> !quake.ref
# CHECK:           %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_0]][2] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.x %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[EXTRACT_REF_1]] : (!quake.ref) -> ()
# CHECK:           %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ALLOCA_0]][1] : (!quake.veq<3>) -> !quake.ref
# CHECK:           quake.apply %[[ARG0]] {{\[}}%[[EXTRACT_REF_0]]] (%[[EXTRACT_REF_2]], %[[EXTRACT_REF_1]]) : (!quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.dealloc %[[ALLOCA_0]] : !quake.veq<3>
# CHECK:           return
# CHECK:         }
