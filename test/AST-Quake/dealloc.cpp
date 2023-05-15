/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --quake-add-deallocs | FileCheck %s

#include <cudaq.h>

__qpu__ void magic_func(int N) {
  {
    cudaq::qreg q(N);
    h(q[0]);
  }
  cudaq::qreg w(N);
  x(w[0]);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_magic_func
// CHECK:           cc.scope {
// CHECK:             %[[VAL_4:.*]] = quake.alloca[%{{.*}} : i64] !quake.veq<?>
// CHECK:             quake.dealloc %[[VAL_4]] : !quake.veq<?>
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = quake.alloca[%{{.*}} : i64] !quake.veq<?>
// CHECK:           quake.dealloc %[[VAL_10]] : !quake.veq<?>
// CHECK:           return
// CHECK:         }

