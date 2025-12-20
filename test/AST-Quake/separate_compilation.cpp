/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_entry_point.
// CHECK-SAME:      () -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_3:.*]] = call @__nvqpp__mlirgen__function_otherKernel.{{.*}}(%{{.*}}) : (!cc.stdvec<i1>) -> i64
