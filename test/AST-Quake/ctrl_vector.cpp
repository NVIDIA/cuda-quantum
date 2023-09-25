/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct lower_ctrl_as_qreg {
  void operator()() __qpu__ {
    cudaq::qreg reg1(4); // group of controls
    cudaq::qreg reg2(2); // some targets

    h(reg1, reg2[0]);
    x(reg1, reg2[1]);

    mz(reg2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__lower_ctrl_as_qreg() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h [%[[VAL_0]]] %[[VAL_2]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_0]]] %[[VAL_3]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_1]] : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }

struct test_two_control_call {
  void operator()() __qpu__ {
    auto lambda = [](cudaq::qubit &qb) __qpu__ {
      h<cudaq::ctrl>(qb);
      x<cudaq::ctrl>(qb);
    };
    cudaq::qreg<4> qs;
    cudaq::qubit qb;
    cudaq::control(lambda, qs, qb);
    mz(qb);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test_two_control_call() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !quake.ref):
// CHECK:             quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           quake.apply @__nvqpp__mlirgen__ZN21test_two_control_callcl[[LAMBDA:.*]]_ [%[[VAL_2]]] %[[VAL_4]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> i1
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:  func.func @__nvqpp__mlirgen__ZN21test_two_control_callcl
// CHECK-SAME:     [[LAMBDA]]_(%{{.*}}: !quake.ref)
// CHECK:          quake.h
// CHECK:          quake.x
// CHECK:          return

struct unmarked_lambda {
  void operator()() __qpu__ {
    auto lambda = [](cudaq::qubit &qb) {
      h<cudaq::ctrl>(qb);
      y<cudaq::ctrl>(qb);
    };
    cudaq::qreg<4> qs;
    cudaq::qubit qb;
    cudaq::control(lambda, qs, qb);
    mz(qb);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__unmarked_lambda() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !quake.ref):
// CHECK:             quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:             quake.y %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           quake.apply %[[VAL_0]] [%[[VAL_2]]] %[[VAL_3]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_3]] : (!quake.ref) -> i1
// CHECK:           return
// CHECK:         }

struct direct_unmarked_lambda {
  void operator()() __qpu__ {
    cudaq::qreg<4> qs;
    cudaq::qubit qb;
    cudaq::control([](cudaq::qubit &qb) {
      h<cudaq::ctrl>(qb);
      y<cudaq::ctrl>(qb);
    }, qs, qb);
    mz(qb);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__direct_unmarked_lambda() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !quake.ref):
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.y %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           quake.apply %[[VAL_2]] [%[[VAL_0]]] %[[VAL_1]] : (!quake.veq<4>, !quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> i1
// CHECK:           return
// CHECK:         }
