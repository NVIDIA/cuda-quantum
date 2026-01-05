/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  // If the swap is working correctly, this will show "101". If it is working
  // incorrectly, it may show something like "011".
  std::cout << "most_probable \"" << result.most_probable() << "\"\n";

  return 0;
}

// CHECK:         tail call void @__quantum__qis__x__body(%[[VAL_0:.*]]* null)
// CHECK:         tail call void @__quantum__qis__x__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__swap__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__mz__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_1:.*]]* writeonly null)
// CHECK:         tail call void @__quantum__qis__mz__body(%[[VAL_0]]* null, %[[VAL_1]]* nonnull writeonly inttoptr (i64 1 to %[[VAL_1]]*))
// CHECK:         tail call void @__quantum__qis__mz__body(%[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*), %[[VAL_1]]* nonnull writeonly inttoptr (i64 2 to %[[VAL_1]]*))
// CHECK:         tail call void @__quantum__rt__result_record_output(%[[VAL_1]]* null, i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303000, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%[[VAL_1]]* nonnull inttoptr (i64 1 to %[[VAL_1]]*), i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303100, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%[[VAL_1]]* nonnull inttoptr (i64 2 to %[[VAL_1]]*), i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303200, i64 0, i64 0))
// CHECK:         ret void
// STDOUT-DAG: { 101:1000 }
// STDOUT-DAG: most_probable "101"
