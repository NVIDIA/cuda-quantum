/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct heisenbergU {
  void operator()(cudaq::qreg<> &q) __qpu__ {
    auto nQubits = q.size();
    for (int step = 0; step < 100; ++step) {
      for (int j = 0; j < nQubits; j++)
        rx(-.01, q[j]);
      for (int i = 0; i < nQubits - 1; i++) {
        cudaq::compute_action([&]() { x<cudaq::ctrl>(q[i], q[i + 1]); },
                              [&]() { rz(-.01, q[i + 1]); });
      }
    }
  }
};

struct ctrlHeisenberg {
  void operator()(int nQubits) __qpu__ {
    cudaq::qubit ctrl1, ctrl2;
    cudaq::qreg q(nQubits);
    cudaq::control(heisenbergU{}, {ctrl1, ctrl2}, q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__heisenbergU(

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenberg(
// CHECK-SAME:        %{{.*}}: i32{{.*}}) attributes
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.concat %[[VAL_2]], %[[VAL_3]] : (!quake.ref, !quake.ref) -> !quake.veq<2>
// CHECK:           quake.apply @__nvqpp__mlirgen__heisenbergU [%[[VAL_8]]] %{{.*}} : (!quake.veq<2>, !quake.veq<?>) -> ()
// CHECK:           return

struct givens {
  void operator()(double lambda, cudaq::qubit &q, cudaq::qubit &r) __qpu__ {
    ry(M_PI_2, q);
    ry(M_PI_2, r);
    z<cudaq::ctrl>(q, r);
    ry(lambda, q);
    ry(-lambda, r);
    z<cudaq::ctrl>(q, r);
    ry(-M_PI_2, q);
    ry(-M_PI_2, r);
  }
};

__qpu__ void qnppx(double theta, cudaq::qubit &q, cudaq::qubit &r,
                   cudaq::qubit &s, cudaq::qubit &t) {
  x<cudaq::ctrl>(r, q);
  x<cudaq::ctrl>(s, t);
  cudaq::control(givens{}, {q, t}, theta, r, s);
  x<cudaq::ctrl>(r, q);
  x<cudaq::ctrl>(s, t);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__givens(

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qnppx
// CHECK:           %[[VAL_7:.*]] = quake.concat %{{.*}}, %{{.*}} : (!quake.ref, !quake.ref) -> !quake.veq<2>
// CHECK:           quake.apply @__nvqpp__mlirgen__givens [%[[VAL_7]]] %{{.*}}, %{{.*}}, %{{.*}} : (!quake.veq<2>, f64, !quake.ref, !quake.ref) -> ()
// CHECK:           return

__qpu__ void magic_func(cudaq::qreg<> &q) {
  auto nQubits = q.size();
  for (int step = 0; step < 100; ++step) {
    for (int j = 0; j < nQubits; j++)
      rx(-.01, q[j]);
    for (int i = 0; i < nQubits - 1; i++) {
      cudaq::compute_action([&]() { x<cudaq::ctrl>(q[i], q[i + 1]); },
                            [&]() { rz(-.01, q[i + 1]); });
    }
  }
}

struct ctrlHeisenbergVersion2 {
  void operator()(int nQubits) __qpu__ {
    cudaq::qubit ctrl1;
    cudaq::qreg q(nQubits);
    cudaq::control(magic_func, ctrl1, q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_magic_func
// CHECK-SAME:      ._Z[[mangle:[^(]*]](

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenbergVersion2(
// CHECK:           quake.apply @__nvqpp__mlirgen__function_magic_func._Z[[mangle]] [%{{.*}}] %{{.*}} : (!quake.ref, !quake.veq<?>) -> ()
// CHECK:           return

__qpu__ void qnppx2(double theta, cudaq::qubit &q, cudaq::qubit &r,
                    cudaq::qubit &s, cudaq::qubit &t) {
  x<cudaq::ctrl>(r, q);
  x<cudaq::ctrl>(s, t);
  cudaq::control(givens{}, {q, !t}, theta, r, s);
  x<cudaq::ctrl>(r, q);
  x<cudaq::ctrl>(s, t);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qnppx2
// CHECK-SAME:       %{{[^:]*}}: f64{{.*}}, %[[VAL_1:.*]]: !quake.ref{{.*}}, %[[VAL_2:.*]]: !quake.ref{{.*}}, %[[VAL_3:.*]]: !quake.ref{{.*}}, %[[VAL_4:.*]]: !quake.ref{{.*}})
// CHECK:           %[[VAL_7:.*]] = quake.concat %[[VAL_1]], %[[VAL_4]] : (!quake.ref, !quake.ref) -> !quake.veq<2>
// CHECK:           quake.x %[[VAL_4]]
// CHECK:           quake.apply @__nvqpp__mlirgen__givens [%[[VAL_7]]] %{{.*}}, %[[VAL_2]], %[[VAL_3]] : (!quake.veq<2>, f64, !quake.ref, !quake.ref) -> ()
// CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           quake.x [%[[VAL_2]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           quake.x [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
