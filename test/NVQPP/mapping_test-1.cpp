/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std %s -o %t --target oqc --emulate && CUDAQ_DUMP_JIT_IR=1 %t 2> %t.code | FileCheck %s && FileCheck --check-prefix=QUAKE %s < %t.code && rm %t.code
// RUN: nvq++ %cpp_std %s -o %t --target iqm --iqm-machine Adonis --mapping-file "%p/../Supplemental/Adonis Variant.txt" --emulate && %t
// RUN: nvq++ %cpp_std --enable-mlir %s -o %t

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

// QUAKE-LABEL: tail call void @__quantum__qis__x__body(%Qubit* null)
// QUAKE:       tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*))
// QUAKE:       tail call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
// QUAKE:       tail call void @__quantum__qis__swap__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
// QUAKE:       tail call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Qubit* nonnull inttoptr (i64 2 to %Qubit*))
// QUAKE:       tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* writeonly null)
// QUAKE:       tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* nonnull writeonly inttoptr (i64 1 to %Result*))
// QUAKE:       tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull writeonly inttoptr (i64 2 to %Result*))
// QUAKE:       tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
// QUAKE:       tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
// QUAKE:       tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 2 to %Result*), i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @cstr.{{.*}}, i64 0, i64 0))
// QUAKE:       ret void

// CHECK-LABEL: most_probable "101"
