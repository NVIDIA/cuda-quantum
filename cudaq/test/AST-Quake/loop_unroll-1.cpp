/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --expand-measurements --unrolling-pipeline | FileCheck %s

#include <cudaq.h>

struct C {
   void operator()() __qpu__ {
      cudaq::qvector r(2);
      cudaq::qubit w;
      auto singleQubit = mz(w);
      auto myRegister = mz(r);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__C()
// CHECK-DAG:       %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK-DAG:       %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] name "singleQubit" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_3]] name "myRegister%[[VAL_0]]" : (!quake.ref) -> !cc.measure_handle
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_5]] name "myRegister%[[VAL_1]]" : (!quake.ref) -> !cc.measure_handle
// CHECK:           return
// CHECK:         }

