/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && CUDAQ_QIR_VERSION_UNDER_DEVELOPMENT=1 CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck --check-prefix CHECK-UNDER-DEVELOPMENT %s
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>

// Test module flags

__qpu__ void kernel_module_flags(int n, std::int32_t m) {
  cudaq::qvector q(n);
  int j = 0;
  for (auto i = 0; i < n; i++) {
    int k = 0;
    if (i > 5)
      k = 1;
    x(q[k]);
    if (mz(q[k])) {
      j = j + 1;
      m = m + m;
    }
  }

  if (j > 5) {
    x(q[0]);
  }
}

int main() {

  // Test module flags
  {
    auto counts = cudaq::sample(kernel_module_flags, 6, 134);
    counts.dump();
  }

  // clang-format off
  // CHECK:  declare i1 @__quantum__qis__read_result__body(%Result*) local_unnamed_addr

  // CHECK:  attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "output_names"="{{.*}}" "qir_profiles"="adaptive_profile" "requiredQubits"="6" "requiredResults"="6" }
  // CHECK:  attributes #1 = { "irreversible" }

  // CHECK:  !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}

  // CHECK:  !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK:  !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK:  !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK:  !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK:  !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK:  !5 = !{i32 1, !"qubit_resetting", i1 true}
  // CHECK:  !6 = !{i32 1, !"classical_ints", i1 true}
  // CHECK:  !7 = !{i32 1, !"classical_floats", i1 true}
  // CHECK:  !8 = !{i32 1, !"classical_fixed_points", i1 true}
  // CHECK:  !9 = !{i32 1, !"user_functions", i1 true}
  // CHECK:  !10 = !{i32 1, !"dynamic_float_args", i1 true}
  // CHECK:  !11 = !{i32 1, !"extern_functions", i1 true}
  // CHECK:  !12 = !{i32 1, !"backwards_branching", i1 true}
  // clang-format on

  // clang-format-off
  // CHECK-UNDER-DEVELOPMENT: ; ModuleID = 'LLVMDialectModule'
  // CHECK-UNDER-DEVELOPMENT: declare i1 @__quantum__rt__read_result(%Result*) local_unnamed_addr
  // CHECK-UNDER-DEVELOPMENT: attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "output_names"="{{.*}}" "qir_profiles"="adaptive_profile" "required_num_qubits"="6" "required_num_results"="6" }
  // CHECK-UNDER-DEVELOPMENT: attributes #1 = { "irreversible" }
  // CHECK-UNDER-DEVELOPMENT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}
  // CHECK-UNDER-DEVELOPMENT: !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK-UNDER-DEVELOPMENT: !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK-UNDER-DEVELOPMENT: !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK-UNDER-DEVELOPMENT: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK-UNDER-DEVELOPMENT: !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK-UNDER-DEVELOPMENT: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
  // CHECK-UNDER-DEVELOPMENT: !6 = !{i32 1, !"backwards_branching", i2 0}
  // clang-format on

 
  return 0;
}
