/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__lower_ctrl_as_qreg
// CHECK-SAME: () attributes {{{.*}}"cudaq-entrypoint"{{.*}}} {
// CHECK:  %[[VAL_0:.*]] = arith.constant 4 : i32
// CHECK:  %[[VAL_2:.*]] = quake.alloca(%{{.*}} : i64) : !quake.qvec<?>
// CHECK:  %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:  %[[VAL_5:.*]] = quake.alloca(%{{.*}} : i64) : !quake.qvec<?>
// CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:  %[[VAL_8:.*]] = quake.qextract %[[VAL_5]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:  quake.h [%[[VAL_2]] : !quake.qvec<?>] (%[[VAL_8]])
// CHECK:  %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:  %[[VAL_11:.*]] = quake.qextract %[[VAL_5]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:  quake.x [%[[VAL_2]] : !quake.qvec<?>] (%[[VAL_11]])
// clang-format on

struct test_two_control_call {
  void operator()() __qpu__ {
    auto lambda = [](cudaq::qubit &qb) {
      h<cudaq::ctrl>(qb);
      x<cudaq::ctrl>(qb);
    };
    cudaq::qreg<4> qs;
    cudaq::qubit qb;
    cudaq::control(lambda, qs, qb);
    mz(qb);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test_two
// CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !quake.qref):
// CHECK:             cc.scope {
// CHECK:               quake.h (%[[VAL_1]])
// CHECK:               quake.x (%[[VAL_1]])
// CHECK:             }
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca(%[[VAL_4]] : i64) : !quake.qvec<4>
// CHECK:           %[[VAL_6:.*]] = quake.alloca : !quake.qref
// CHECK:           quake.apply @__nvqpp__mlirgen__{{.*}}test_two_control_call{{.*}}[%[[VAL_5]] : !quake.qvec<4>] %[[VAL_6]] : (!quake.qref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.mz(%[[VAL_6]] : !quake.qref) : i1
// CHECK:           return
// CHECK:         }

