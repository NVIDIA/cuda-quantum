/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ -v %s -o %t --target quantinuum --emulate && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck --check-prefixes=CHECK,QIR_ADAPTIVE %s
// RUN: nvq++ -v %s -o %t --target ionq --emulate && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck --check-prefixes=CHECK,IONQ %s
// RUN: if %qci_avail; then nvq++ -v %s -o %t --target qci --emulate && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck --check-prefixes=CHECK,QIR_ADAPTIVE %s; fi
// RUN: nvq++ --enable-mlir %s -o %t
// clang-format on

// Note: iqm (and others) that don't use QIR should not be included in this test.

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

// clang-format off
// QIR_ADAPTIVE: @cstr.[[ADDRESS:[A-Z0-9]+]] = private constant [14 x i8] c"measureResult\00"
// CHECK-LABEL: define void @__nvqpp__mlirgen__function_qir_test.
// CHECK-SAME:    () local_unnamed_addr #[[ATTR_1:[0-9]+]] {
// QIR_ADAPTIVE:         call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @cstr.[[ADDRESS]], i64 0, i64 0))
// IONQ:         tail call void @__quantum__qis__x__body(
// CHECK:     attributes #[[ATTR_1]] = { "entry_point" {{.*}}"qir_profiles"="{{.*}}_profile" "requiredQubits"="1" "requiredResults"="1" }
