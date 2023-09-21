/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void init_state(cudaq::qreg<> &qubits, double theta) {}

__qpu__ void reflect_uni(cudaq::qreg<> &ctrls, cudaq::qreg<> &qubits,
                         double theta) {
  cudaq::adjoint(init_state, qubits, theta);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_init_state.
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: f64)
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_reflect_uni.
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.veq<?>, %[[VAL_2:.*]]: f64)
// CHECK:           %[[VAL_3:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f64>
// CHECK:           quake.apply<adj> @__nvqpp__mlirgen__function_init_state.{{.*}} %[[VAL_1]], %[[VAL_5]] : (!quake.veq<?>, f64) -> ()
// CHECK:           return
// CHECK:         }

