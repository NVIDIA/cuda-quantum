/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --pass-pipeline='builtin.module(expand-measurements,canonicalize,cc-loop-unroll,canonicalize)' | FileCheck %s

#include <cudaq.h>

struct C {
   void operator()() __qpu__ {
      cudaq::qreg r(2);
      mz(r);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__C()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca : !quake.qvec<2>
// CHECK-DAG:       %[[VAL_4:.*]] = llvm.alloca %[[VAL_0]] x i1 : (i64) -> !llvm.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_2]]] : !quake.qvec<2>[index] -> !quake.qref
// CHECK:           %[[VAL_6:.*]] = quake.mz(%[[VAL_5]] : !quake.qref) : i1
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_4]] : !llvm.ptr<i1>
// CHECK:           %[[VAL_7:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_1]]] : !quake.qvec<2>[index] -> !quake.qref
// CHECK:           %[[VAL_8:.*]] = quake.mz(%[[VAL_7]] : !quake.qref) : i1
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_4]][1] : (!llvm.ptr<i1>) -> !llvm.ptr<i1>
// CHECK:           llvm.store %[[VAL_8]], %[[VAL_9]] : !llvm.ptr<i1>
// CHECK:           return
// CHECK:         }

