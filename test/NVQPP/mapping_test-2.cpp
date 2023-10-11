/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -v %s -o %basename_t.x --target oqc --emulate && OQC_EMAIL=0 OQC_PASSWORD=0 CUDAQ_DUMP_JIT_IR=1 ./%basename_t.x |& FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qvector q(3);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[0], q[2]); // requires a swap
  mz(q);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  return 0;
}

// CHECK:         tail call void @__quantum__qis__h__body(%[[VAL_0:.*]]* null)
// CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__swap__body(%[[VAL_0]]* null, %[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__cnot__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*))
// CHECK:         tail call void @__quantum__qis__mz__body(%[[VAL_0]]* nonnull inttoptr (i64 1 to %[[VAL_0]]*), %[[VAL_1:.*]]* writeonly null)
// CHECK:         tail call void @__quantum__qis__mz__body(%[[VAL_0]]* null, %[[VAL_1]]* nonnull writeonly inttoptr (i64 1 to %[[VAL_1]]*))
// CHECK:         tail call void @__quantum__qis__mz__body(%[[VAL_0]]* nonnull inttoptr (i64 2 to %[[VAL_0]]*), %[[VAL_1]]* nonnull writeonly inttoptr (i64 2 to %[[VAL_1]]*))
// CHECK:         tail call void @__quantum__rt__result_record_output(%[[VAL_1]]* nonnull inttoptr (i64 1 to %[[VAL_1]]*), i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303100, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%[[VAL_1]]* null, i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303000, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%[[VAL_1]]* nonnull inttoptr (i64 2 to %[[VAL_1]]*), i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303200, i64 0, i64 0))
// CHECK:         ret void
