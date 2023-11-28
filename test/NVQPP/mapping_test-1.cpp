/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -v %s -o %t --target oqc --emulate && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qubit q0, q1, q2;
  x(q0);
  x(q1);
  x<cudaq::ctrl>(q0, q1);
  x<cudaq::ctrl>(q0, q2); // requires a swap(q0,q1)
  auto q0result = mz(q0);
  auto q1result = mz(q1);
  auto q2result = mz(q2);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  // If the swap is working correctly, this will show "101". If it is working
  // incorrectly, it may show something like "011".
  std::cout << "most_probable \"" << result.most_probable() << "\"\n";

  return 0;
}

// CHECK:         tail call void @__quantum__qis__x__body(%Qubit* null)
// CHECK:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
// CHECK:         tail call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
// CHECK:         tail call void @__quantum__qis__swap__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
// CHECK:         tail call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
// CHECK:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* writeonly null)
// CHECK:         tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull writeonly inttoptr (i64 1 to %Result*))
// CHECK:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull writeonly inttoptr (i64 2 to %Result*))
// CHECK:         tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 2 to %Result*), i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
// CHECK:         ret void
// CHECK:         most_probable "101"
