# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: CUDAQ_DUMP_JIT_IR=1 PYTHONPATH=../../.. python3 %s --target quantinuum --emulate |& FileCheck %s

import cudaq
import numpy as np

# Test hasMultipleTargetBranching flag
# TODO: add test after the support for switch statements is added.


# Test multiple_return_points flag
# NOTE: we create a common return with a phi node, not currently possible to
# produce a `True` value for the multiple_return_points flag
def test_return():

    @cudaq.kernel
    def kernel(n: int) -> int:
        q = cudaq.qvector(n)
        if mz(q[0]):
            x(q[0])
            return 1
        return 2

    print(cudaq.sample(kernel, 2))


test_return()

# CHECK: ; ModuleID = 'LLVMDialectModule'
# CHECK: {{.*}}
# CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

# CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
# CHECK: !6 = !{i32 1, !"backwards_branching", i2 0}


# Test int_computations flag
# TODO: add cudaq.run tests using runtime output functions
def test_int_computations():

    @cudaq.kernel
    def kernel(n: int, m: np.int32):
        q = cudaq.qvector(n)
        j = 0
        for i in range(10):
            k = 0
            if i > 5:
                k = 1
            x(q[k])
            if mz(q[k]):
                j = j + 1
                m = m + m

        if j > 5:
            x(q[0])

    cudaq.sample(kernel, 2, 134)


test_int_computations()

# CHECK: {{.*}}
# CHECK: ; ModuleID = 'LLVMDialectModule'
# CHECK: {{.*}}
# CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

# CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# CHECK: !5 = !{i32 1, !"int_computations", [6 x i8] c"i1,i64"}
# CHECK: !6 = !{i32 1, !"backwards_branching", i2 0}

# Test int_computations and float_computations flags
# TODO: add cudaq.run tests using runtime output functions
# def test_float_computations():

#     @cudaq.kernel
#     def kernel(n: int, m: np.int32):
#         q = cudaq.qvector(n)
#         j = 0
#         jf = 1.2
#         for i in range(10):
#             k = 0
#             if i > 5:
#                 k = 1
#             x(q[k])
#             if mz(q[k]):
#                 j = j + 1
#                 m = m + m
#                 jf = jf + jf

#         if jf > 3 and j > 5:
#             x(q[0])

#     cudaq.sample(kernel, 2, 134)

# test_float_computations()

# XHECK: {{.*}}
# XHECK: ; ModuleID = 'LLVMDialectModule'
# XHECK: {{.*}}
# XHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}

# XHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# XHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# XHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# XHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# XHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# XHECK: !5 = !{i32 1, !"int_computations", [6 x i8] c"i1,i64"}
# XHECK: !6 = !{i32 1, !"float_computations", [3 x i8] c"f64"}
# XHECK: !7 = !{i32 1, !"backwards_branching", i2 0}

# Test backwards_branching flag


def test_iteration_loop():

    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        i = 0
        j = 0
        while i < 10000 - 1:
            cx(q[j], q[j + 1])
            i = i + 1
            j = j + 1
            if j >= n - 1:
                j = 0

    print(cudaq.sample(kernel, 2))


test_iteration_loop()

# CHECK: {{.*}}
# CHECK: ; ModuleID = 'LLVMDialectModule'
# CHECK: {{.*}}
# CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

# CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
# CHECK: !6 = !{i32 1, !"backwards_branching", i2 1}


def test_conditionally_terminating_loops():

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        t = False
        while not t:
            x(q)
            t = mz(q)

    cudaq.sample(kernel)


test_conditionally_terminating_loops()

# CHECK: {{.*}}
# CHECK: ; ModuleID = 'LLVMDialectModule'
# CHECK: {{.*}}
# CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

# CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# NOTE: -2 is "10" bitstring base 2
# CHECK: !5 = !{i32 1, !"backwards_branching", i2 -2}


def test_iteration_and_conditionally_terminating_loop():

    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        i = 0
        j = 0
        # Use large number of iterations to prevent unrolling.
        while i < 1025:
            cx(q[j], q[j + 1])
            i = i + 1
            j = j + 1
            if j >= n - 1:
                j = 0

        qbit = cudaq.qubit()
        t = False
        while not t:
            h(qbit)
            t = mz(qbit)

    print(cudaq.sample(kernel, 2))


test_iteration_and_conditionally_terminating_loop()

# CHECK: {{.*}}
# CHECK: ; ModuleID = 'LLVMDialectModule'
# CHECK: {{.*}}
# CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

# CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
# NOTE: -1 is "11" bitstring base 2
# CHECK: !6 = !{i32 1, !"backwards_branching", i2 -1}
