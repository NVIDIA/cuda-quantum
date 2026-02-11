/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --add-dealloc --lower-to-cfg | cudaq-translate --convert-to=qir-base -o - | FileCheck %s
// clang-format on

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    x<cudaq::ctrl>(q[0], q[1]);
    h(q[0]);

    // This scope block is intentionally blank and is used for robustness
    // testing.
    {}

    auto b0 = mz(q[0]);
    auto b1 = mz(q[1]);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel()
// CHECK:         tail call void @__quantum__qis__mz__body(ptr null, ptr null)
// CHECK:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 1 to ptr))
// CHECK:         tail call void @__quantum__rt__result_record_output(ptr null, ptr nonnull @cstr.623000)
// CHECK:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull @cstr.623100)
// clang-format on
