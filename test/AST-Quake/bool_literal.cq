/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s


#include <cudaq.h>

struct testBoolLiteral {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    h(q);    
    bool bit = false;
    bit = mz(q);
    return bit;
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testBoolLiteral() -> i1 attributes
// CHECK:           %[[VAL_0:.*]] = arith.constant false
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> i1
// CHECK:           cc.store %[[VAL_3]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           return %[[VAL_4]] : i1

