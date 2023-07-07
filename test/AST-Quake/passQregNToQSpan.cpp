/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include "cudaq.h"

__qpu__ void mcx(cudaq::qspan<> qubits) {

}

struct entry {
 void operator()() __qpu__ {
    cudaq::qreg<3> q;
    mcx(q);
 }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_mcx.
// CHECK-SAME:      (%[[VAL_0:.*]]: !quake.veq<?>{{.*}}) attributes {
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__entry() attributes {
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_2:.*]] = quake.relax_size %[[VAL_1]] : (!quake.veq<3>) -> !quake.veq<?>
// CHECK:           call @__nvqpp__mlirgen__function_mcx._Z3mcxN5cudaq5qspan{{.*}}(%[[VAL_2]]) : (!quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

