/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
// clang-format on

// An initializer list of quantum references always lowers to a !quake.veq,
// even when the list has exactly one element. It must not be collapsed to the
// bare !quake.ref of that element.

#include <cudaq.h>

struct flip {
  void operator()(cudaq::qubit &t) __qpu__ { x(t); }
};

__qpu__ void single_qubit(cudaq::qubit &c, cudaq::qubit &t) {
  cudaq::control(flip{}, {c}, t);
}

__qpu__ void single_extracted(cudaq::qvector<> &v, cudaq::qubit &t) {
  cudaq::control(flip{}, {v[0]}, t);
}

__qpu__ void two_qubits(cudaq::qubit &c, cudaq::qubit &d, cudaq::qubit &t) {
  cudaq::control(flip{}, {c, d}, t);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_single_qubit
// CHECK-SAME:      (%[[ARG0:.*]]: !quake.ref, %[[ARG1:.*]]: !quake.ref)
// CHECK:           %[[V:.*]] = quake.concat %[[ARG0]] : (!quake.ref) -> !quake.veq<1>
// CHECK:           quake.apply @__nvqpp__mlirgen__flip [%[[V]]] (%[[ARG1]]) : (!quake.veq<1>, !quake.ref) -> ()

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_single_extracted
// CHECK-SAME:      (%[[ARG0:.*]]: !quake.veq<?>, %[[ARG1:.*]]: !quake.ref)
// CHECK:           %[[R:.*]] = quake.extract_ref %[[ARG0]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           %[[V:.*]] = quake.concat %[[R]] : (!quake.ref) -> !quake.veq<1>
// CHECK:           quake.apply @__nvqpp__mlirgen__flip [%[[V]]] (%[[ARG1]]) : (!quake.veq<1>, !quake.ref) -> ()

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_two_qubits
// CHECK:           %[[V:.*]] = quake.concat %{{.*}}, %{{.*}} : (!quake.ref, !quake.ref) -> !quake.veq<2>
// CHECK:           quake.apply @__nvqpp__mlirgen__flip [%[[V]]] (%{{.*}}) : (!quake.veq<2>, !quake.ref) -> ()

// The veq must survive all the way through lowering to QIR.
// QIR-LABEL: define void @__nvqpp__mlirgen__function_single_qubit
// QIR:         @generalizedInvokeWithRotationsControlsTargets({{.*}}@__quantum__qis__x__ctl
// clang-format on
