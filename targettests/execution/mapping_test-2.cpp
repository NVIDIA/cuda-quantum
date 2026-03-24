/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t --target oqc --emulate && CUDAQ_DUMP_JIT_IR=1 %t 2> %t.txt | FileCheck --check-prefix=STDOUT %s && FileCheck %s < %t.txt
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qvector q(3);
  x(q[0]);
  x(q[1]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[0], q[2]); // requires a swap(q0,q1)
  auto result = mz(q);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  // If the swap is working correctly, this will show "101". If it is working
  // incorrectly, it may show something like "011".
  std::cout << "most_probable \"" << result.most_probable() << "\"\n";

  return 0;
}

// CHECK:         tail call void @__quantum__qis__x__body(ptr null)
// CHECK:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 1 to ptr))
// CHECK:         tail call void @__quantum__qis__cnot__body(ptr null, ptr nonnull inttoptr (i64 1 to ptr))
// CHECK:         tail call void @__quantum__qis__swap__body(ptr null, ptr nonnull inttoptr (i64 1 to ptr))
// CHECK:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 2 to ptr))
// CHECK:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr writeonly null)
// CHECK:         tail call void @__quantum__qis__mz__body(ptr null, ptr nonnull writeonly inttoptr (i64 1 to ptr))
// CHECK:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull writeonly inttoptr (i64 2 to ptr))
// CHECK:         tail call void @__quantum__rt__array_record_output(i64 3, ptr nonnull @cstr.61727261793C6931207820333E00)
// CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull null, ptr nonnull @cstr.726573756C74253000)
// CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull @cstr.726573756C74253100)
// CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull @cstr.726573756C74253200)
// CHECK:         ret void
// STDOUT-DAG: __global__ : { 101:1000 }
// STDOUT-DAG: result%0 : { 1:1000 }
// STDOUT-DAG: result%1 : { 0:1000 }
// STDOUT-DAG: result%2 : { 1:1000 }
// STDOUT-DAG: most_probable "101"
