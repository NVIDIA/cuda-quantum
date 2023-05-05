/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

// Verifies that cphase op produces the same IR as r1<cudaq::ctrl>
struct CphaseOp {
  void operator()() __qpu__ {
    cudaq::qubit q1, q2;
    cphase(1.234, q1, q2);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CphaseOp() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
// CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : f64
// CHECK:           quake.r1 [%[[VAL_0]] : !quake.qref] |%[[VAL_2]] : f64|(%[[VAL_1]])
// CHECK:           return
// CHECK:         }
// clang-format on

struct CtrlR1 {
  void operator()() __qpu__ {
    cudaq::qubit q1, q2;
    r1<cudaq::ctrl>(1.234, q1, q2);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__CtrlR1() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
// CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
// CHECK:           %[[VAL_2:.*]] = arith.constant {{.*}} : f64
// CHECK:           quake.r1 [%[[VAL_0]] : !quake.qref] |%[[VAL_2]] : f64|(%[[VAL_1]])
// CHECK:           return
// CHECK:         }
// clang-format on
