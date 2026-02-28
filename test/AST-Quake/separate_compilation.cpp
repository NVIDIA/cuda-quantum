/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ uint64_t otherKernel(std::vector<cudaq::measure_result> &x);

__qpu__ uint64_t test_entry_point() {
  cudaq::qvector q(5);
  auto results = cudaq::mz(q);
  return otherKernel(results);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_entry_point._Z16test_entry_pointv() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<5>
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "results" : (!quake.veq<5>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_2:.*]] = call @{{.*otherKernel.*}}(%[[VAL_1]]) : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:           return %[[VAL_2]] : i64
// CHECK:         }
