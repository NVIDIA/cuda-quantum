/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ -v %s -o %basename_t.x --target quantinuum --emulate && CUDAQ_DUMP_JIT_IR=1 ./%basename_t.x |& FileCheck %s
// RUN: nvq++ -v %s -o %basename_t.x --target ionq --emulate && IONQ_API_KEY=0 CUDAQ_DUMP_JIT_IR=1 ./%basename_t.x |& FileCheck --check-prefix IONQ %s
// Note: iqm not currently tested because it does not currently use QIR

#include <cudaq.h>
#include <iostream>

__qpu__ void qir_test() {
  cudaq::qubit q;
  x(q);
  auto measureResult = mz(q);
};

int main() {
  auto result = cudaq::sample(1000, qir_test);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// CHECK: @cstr.[[ADDRESS:[A-Z0-9]+]] = private constant [14 x i8] c"measureResult\00"
// CHECK-DAG: declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) local_unnamed_addr #[[ATTR_0:[0-9]+]]
// CHECK-DAG: define void @__nvqpp__mlirgen__function_qir_test.{{.*}}() local_unnamed_addr #[[ATTR_1:[0-9]+]]
// CHECK: call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @cstr.[[ADDRESS]], i64 0, i64 0))
// CHECK-DAG: attributes #[[ATTR_0]] = { "irreversible" }
// CHECK-DAG: attributes #[[ATTR_1]] = { "entry_point" {{.*}} "qir_profiles"="base_profile" "requiredQubits"="1" "requiredResults"="1" }
// CHECK-DAG: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK-DAG: !0 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK-DAG: !1 = !{i32 1, !"qir_major_version", i32 1}
// CHECK-DAG: !2 = !{i32 7, !"qir_minor_version", i32 0}
// CHECK-DAG: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// CHECK-DAG: !4 = !{i32 1, !"dynamic_result_management", i1 false}

// IONQ: define void @__nvqpp__mlirgen__function_qir_test.{{.*}}() local_unnamed_addr #[[ATTR_1:[0-9]+]]
// IONQ-DAG: attributes #[[ATTR_1]] = { "entry_point" {{.*}} "output_names"={{.*}} "qir_profiles"="base_profile" "requiredQubits"="1" "requiredResults"="1" }
// IONQ-DAG: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// IONQ-DAG: !0 = !{i32 2, !"Debug Info Version", i32 3}
// IONQ-DAG: !1 = !{i32 1, !"qir_major_version", i32 1}
// IONQ-DAG: !2 = !{i32 7, !"qir_minor_version", i32 0}
// IONQ-DAG: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// IONQ-DAG: !4 = !{i32 1, !"dynamic_result_management", i1 false}
