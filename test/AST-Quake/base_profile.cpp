/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %cpp_std %s | cudaq-opt --lower-to-cfg | cudaq-translate --convert-to=qir-base | FileCheck %s
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
// CHECK-DAG:     %[[VAL_0:.*]] = tail call target("qir#Qubit") @llvm.qir.i64ToQubit(i64 1)
// CHECK-DAG:     %[[VAL_1:.*]] = tail call target("qir#Qubit") @llvm.qir.i64ToQubit(i64 2)
// CHECK-DAG:     %[[VAL_2:.*]] = tail call target("qir#Qubit") @llvm.qir.i64ToQubit(i64 0)
// CHECK-DAG:     %[[VAL_3:.*]] = tail call target("qir#Result") @llvm.qir.i64ToResult(i64 0)
// CHECK:         tail call void @__quantum__qis__mz__body(target("qir#Qubit") %[[VAL_2]], target("qir#Result") %[[VAL_3]])
// CHECK:         %[[VAL_4:.*]] = tail call target("qir#Result") @llvm.qir.i64ToResult(i64 1)
// CHECK:         tail call void @__quantum__qis__mz__body(target("qir#Qubit") %[[VAL_0]], target("qir#Result") %[[VAL_4]])
// CHECK:         %[[VAL_5:.*]] = tail call target("qir#Result") @llvm.qir.i64ToResult(i64 0)
// CHECK:         tail call void @__quantum__rt__result_record_output(target("qir#Result") %[[VAL_5]], ptr nonnull @cstr.{{.*}})
// CHECK:         %[[VAL_6:.*]] = tail call target("qir#Result") @llvm.qir.i64ToResult(i64 1)
// CHECK:         tail call void @__quantum__rt__result_record_output(target("qir#Result") %[[VAL_6]], ptr nonnull @cstr.{{.*}})
