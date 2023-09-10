/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --pass-pipeline='builtin.module(expand-measurements,unrolling-pipeline)' | FileCheck %s

#include <cudaq.h>

struct C {
   void operator()() __qpu__ {
      cudaq::qreg r(2);
      cudaq::qubit w;
      auto singleQubit = mz(w);
      auto myRegister = mz(r);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__C()
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK-DAG:       %[[VAL_10:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_11:.*]] = quake.mz %[[VAL_10]] name "singleQubit" : (!quake.ref) -> i1
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.array<i1 x 2>
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_5]] name "myRegister%0" : (!quake.ref) -> i1
// CHECK:           cc.store %[[VAL_6]], %{{.*}} : !cc.ptr<i1>
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_7]] name "myRegister%1" : (!quake.ref) -> i1
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<i1 x 2>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<i1>
// CHECK:           return
// CHECK:         }

