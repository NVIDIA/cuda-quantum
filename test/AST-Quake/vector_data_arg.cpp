/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

void storeHere(int *ptr, int size) { printf("Hi %d\n", ptr[0]); }

__qpu__ void vector_data_as_argument() {
  std::vector<int> ten(10);
  ten[0] = 10;
  storeHere(ten.data(), 10);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_vector_data_as_argument._Z23vector_data_as_argumentv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.array<i32 x 10>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i32 x 10>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<i32 x 10>>) -> !cc.ptr<i32>
// CHECK:           call @_Z9storeHerePii(%[[VAL_3]], %[[VAL_0]]) : (!cc.ptr<i32>, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK:         func.func private @_Z9storeHerePii(!cc.ptr<i32>, i32)

