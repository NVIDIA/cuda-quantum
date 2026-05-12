/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t --target oqc --emulate && CUDAQ_DUMP_JIT_IR=1 %t 2> %t.txt | FileCheck %s &&  FileCheck --check-prefix=QUAKE %s < %t.txt; status=$?; rm -f %t.txt; exit "$status"
// RUN: nvq++ %s -o %t --target iqm --emulate --mapping-file "%iqm_tests_dir/Crystal_5.txt" && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qubit q0, q1, q2;
  x(q0);
  x(q1);
  x<cudaq::ctrl>(q0, q1);
  x<cudaq::ctrl>(q0, q2); // requires a swap(q0,q1)
  mz(q0);
  mz(q1);
  mz(q2);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  // If the swap is working correctly, this will show "101". If it is working
  // incorrectly, it may show something like "011".
  std::cout << "most_probable \"" << result.most_probable() << "\"\n";

  return 0;
}

// clang-format off
// QUAKE-LABEL: tail call void @__quantum__qis__x__body(ptr null)
// QUAKE:       tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 1 to ptr))
// QUAKE:       tail call void @__quantum__qis__cnot__body(ptr null, ptr nonnull inttoptr (i64 1 to ptr))
// QUAKE:       tail call void @__quantum__qis__swap__body(ptr null, ptr nonnull inttoptr (i64 1 to ptr))
// QUAKE:       tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 2 to ptr))
// QUAKE:       tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr writeonly null)
// QUAKE:       tail call void @__quantum__qis__mz__body(ptr null, ptr nonnull writeonly inttoptr (i64 1 to ptr))
// QUAKE:       tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull writeonly inttoptr (i64 2 to ptr))
// QUAKE:       tail call void @__quantum__rt__array_record_output(i64 3, ptr nonnull @cstr.{{.*}})
// QUAKE:       tail call void @__quantum__rt__result_record_output(ptr nonnull null, ptr nonnull @cstr.{{.*}})
// QUAKE:       tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull @cstr.{{.*}})
// QUAKE:       tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull @cstr.{{.*}})
// QUAKE:       ret void

// CHECK-LABEL: most_probable "101"
// clang-format on
