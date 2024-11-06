/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

namespace cudaq {
__qpu__ void callMe(int i) {}
__qpu__ void kernel() { cudaq::callMe(5); }
} // namespace cudaq

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_callMe._ZN5cudaq6callMeEi(
// CHECK-SAME:                                                                    %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._ZN5cudaq6kernelEv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK:           call @__nvqpp__mlirgen__function_callMe._ZN5cudaq6callMeEi(%[[VAL_0]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN5cudaq6callMeEi(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32) attributes {no_this} {
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN5cudaq6kernelEv() attributes {no_this} {
// CHECK:           return
// CHECK:         }