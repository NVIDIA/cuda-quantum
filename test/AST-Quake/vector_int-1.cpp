/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt -kernel-execution | FileCheck %s

#include <cudaq.h>

__qpu__ std::vector<int> doubleDeckerBus() {
  std::vector<int> ii(2);
  ii[0] = 2;
  return ii;
}

__qpu__ void touringLondon() {
  auto ii = doubleDeckerBus();
  cudaq::qvector q(ii[0]);
  return;
}

// CHECK-LABEL:  func.func @__nvqpp__mlirgen__function_doubleDeckerBus._Z15doubleDeckerBusv() -> !cc.stdvec<i32> attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<i32 x 2>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_6]], %[[VAL_1]], %[[VAL_2]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_1]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i32>
// CHECK:           return %[[VAL_8]] : !cc.stdvec<i32>
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_touringLondon._Z13touringLondonv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = call @__nvqpp__mlirgen__function_doubleDeckerBus._Z15doubleDeckerBusv() : () -> !cc.stdvec<i32>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<i32>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i32 x ?>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.cast signed %[[VAL_7]] : (i32) -> i64
// CHECK:           %[[VAL_9:.*]] = quake.alloca !quake.veq<?>[%[[VAL_8]] : i64]
// CHECK:           return
// CHECK:         }
