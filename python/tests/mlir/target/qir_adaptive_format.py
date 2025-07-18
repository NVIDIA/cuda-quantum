# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: CUDAQ_DUMP_JIT_IR=1 PYTHONPATH=../../.. python3 %s --target quantinuum --emulate |& FileCheck %s
# RUN: CUDAQ_QIR_VERSION_UNDER_DEVELOPMENT=1 CUDAQ_DUMP_JIT_IR=1 PYTHONPATH=../../.. python3 %s --target quantinuum --emulate |& FileCheck --check-prefix CHECK-UNDER-DEVELOPMENT %s

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
# CHECK: declare i1 @__quantum__qis__read_result__body(%Result*) local_unnamed_addr
# CHECK: attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "output_names"="{{.*}}" "qir_profiles"="adaptive_profile" "requiredQubits"="2" "requiredResults"="10" }
# CHECK: attributes #1 = { "irreversible" }
# CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
# CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# CHECK: !5 = !{i32 1, !"qubit_resetting", i1 true}
# CHECK: !6 = !{i32 1, !"classical_ints", i1 true}
# CHECK: !7 = !{i32 1, !"classical_floats", i1 true}
# CHECK: !8 = !{i32 1, !"classical_fixed_points", i1 true}
# CHECK: !9 = !{i32 1, !"user_functions", i1 true}
# CHECK: !10 = !{i32 1, !"dynamic_float_args", i1 true}
# CHECK: !11 = !{i32 1, !"extern_functions", i1 true}
# CHECK: !12 = !{i32 1, !"backwards_branching", i1 true}

# CHECK-UNDER-DEVELOPMENT: ; ModuleID = 'LLVMDialectModule'
# CHECK-UNDER-DEVELOPMENT: declare i1 @__quantum__qis__read_result__body(%Result*) local_unnamed_addr
# CHECK-UNDER-DEVELOPMENT: attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "output_names"="{{.*}}" "qir_profiles"="adaptive_profile" "required_num_qubits"="2" "required_num_results"="10" }
# CHECK-UNDER-DEVELOPMENT: attributes #1 = { "irreversible" }
# CHECK-UNDER-DEVELOPMENT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
# CHECK-UNDER-DEVELOPMENT: !0 = !{i32 2, !"Debug Info Version", i32 3}
# CHECK-UNDER-DEVELOPMENT: !1 = !{i32 1, !"qir_major_version", i32 1}
# CHECK-UNDER-DEVELOPMENT: !2 = !{i32 7, !"qir_minor_version", i32 0}
# CHECK-UNDER-DEVELOPMENT: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
# CHECK-UNDER-DEVELOPMENT: !4 = !{i32 1, !"dynamic_result_management", i1 false}
# CHECK-UNDER-DEVELOPMENT: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
# CHECK-UNDER-DEVELOPMENT: !6 = !{i32 1, !"backwards_branching", i2 0}
