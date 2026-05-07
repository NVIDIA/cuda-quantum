/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck %s
// clang-format on

#include <cudaq.h>
#include "cudaq/qis/qubit_qis.h"

void external_call_to_keep_result(std::int64_t results_int) {}

struct kernel {
  void operator()() __qpu__ {
    cudaq::qvector q(16);
    int64_t results_int = cudaq::to_integer(mz(q));
    external_call_to_keep_result(results_int);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel()
// CHECK-NOT: llvm.vector

