# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq


def test_control_list_rotation():
    """Tests the controlled rotation gates, provided a list of controls."""
    kernel, value = cudaq.make_kernel(float)
    target = kernel.qalloc()
    q1 = kernel.qalloc()
    q2 = kernel.qalloc()

    controls = [q1, q2]
    controls_reversed = [q2, q1]

    kernel.crx(value, controls, target)
    kernel.crx(1.0, controls_reversed, target)

    kernel.cry(value, controls_reversed, target)
    kernel.cry(2.0, controls, target)

    kernel.crz(value, controls, target)
    kernel.crz(3.0, controls_reversed, target)

    kernel.cr1(value, controls_reversed, target)
    kernel.cr1(4.0, controls, target)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_367535629127(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 4.000000e+00 : f64
# CHECK:           %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f64
# CHECK:           %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f64
# CHECK:           %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.ref
# CHECK:           quake.rx (%[[VAL_0]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rx (%[[VAL_4]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_0]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_3]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rz (%[[VAL_0]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rz (%[[VAL_2]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.r1 (%[[VAL_0]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.r1 (%[[VAL_1]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
