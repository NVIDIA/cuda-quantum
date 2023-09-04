/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct NegationOperatorTest {
  void operator()() __qpu__ {
    cudaq::qreg qr(3);
    x<cudaq::ctrl>(!qr[0], qr[1], qr[2]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__NegationOperatorTest()
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_1]], %[[VAL_2]] neg [true, false]] %[[VAL_3]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           return

