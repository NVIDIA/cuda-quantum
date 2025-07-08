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


# Test module flags
def test_module_flags():

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


test_module_flags()

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
