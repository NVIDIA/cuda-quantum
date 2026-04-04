/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Simple test using a type inferenced return value type.

#include <cudaq.h>

struct ak2 {
  auto operator()() __qpu__ {
    cudaq::qarray<5> q;
    h(q);
    return mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ak2
// CHECK-SAME:      () -> !quake.measurements<?> attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<5>
// CHECK:           cc.loop while
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<5>) -> !quake.measurements<5>
// CHECK:           %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !quake.measurements<5> to !quake.measurements<?>
// CHECK:           return %[[VAL_2]] : !quake.measurements<?>
// CHECK:         }
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1

