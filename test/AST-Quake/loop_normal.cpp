/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --cc-loop-normalize --canonicalize | FileCheck %s

#include <cudaq.h>

__qpu__ void foo1() {
  cudaq::qubit q;
  for (int i = 0; i < 10; i += 2)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo1
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i32)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32):
// CHECK:             quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_7]] : i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_1]] : i32
// CHECK:             cc.continue %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

__qpu__ void foo2() {
  cudaq::qubit q;
  for (int i = 15; i > 0; i -= 3)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo2
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i32)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32):
// CHECK:             quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_7]] : i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_1]] : i32
// CHECK:             cc.continue %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

__qpu__ void foo3() {
  cudaq::qreg q(10);
  for (int i = 0; i < 10; i += 2)
    x(q[i]);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo3
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_3]]) -> (i32)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:             %[[VAL_9:.*]] = arith.muli %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_10:.*]] = arith.extsi %[[VAL_9]] : i32 to i64
// CHECK:             %[[VAL_11:.*]] = quake.extract_ref %{{.*}}[%[[VAL_10]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:             cc.continue %[[VAL_8]] : i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i32):
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i32
// CHECK:             cc.continue %[[VAL_13]] : i32

__qpu__ void foo4() {
  cudaq::qreg q(10);
  for (int i = 10; i > 0; i -= 2)
    x(q[i-1]);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo4
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 9 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_4]]) -> (i32)) {
// CHECK:             %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32):
// CHECK:             %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_0]], %[[VAL_10]]  : i32
// CHECK:             %[[VAL_12:.*]] = arith.extsi %[[VAL_11]] : i32 to i64
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %{{.*}}[%[[VAL_12]]] : (!quake.veq<10>, i64) -> !quake.ref
// CHECK:           ^bb0(%[[VAL_14:.*]]: i32):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_2]] : i32

__qpu__ void foo5() {
  cudaq::qubit q;
  for (int i = 0; i < 9; i += 2)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo5
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i32)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32):
// CHECK:             quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_7]] : i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_1]] : i32
// CHECK:             cc.continue %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

__qpu__ void foo6() {
  cudaq::qubit q;
  for (int i = -2; i < 16; i += 4)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo6
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i32)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32):
// CHECK:             quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_7]] : i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_1]] : i32
// CHECK:             cc.continue %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

__qpu__ void negative1() {
  cudaq::qubit q;
  for (int i = 5; i < 5; i += 4)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_negative1
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.loop while ((%[[VAL_4:.*]] = %[[VAL_0]]) -> (i32)) {
// CHECK:             %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_5]](%[[VAL_4]] : i32)

__qpu__ void negative2() {
  cudaq::qubit q;
  for (int i = 5; i <= 4; i += 32)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_negative2
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.loop while ((%[[VAL_4:.*]] = %[[VAL_0]]) -> (i32)) {
// CHECK:             %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_5]](%[[VAL_4]] : i32)

__qpu__ void negative3() {
  cudaq::qubit q;
  for (int i = -10; i < -1; i += -2)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_negative3
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = cc.loop while ((%[[VAL_4:.*]] = %[[VAL_0]]) -> (i32)) {
// CHECK:             %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_0]] : i32
// CHECK:             cc.condition %[[VAL_5]](%[[VAL_4]] : i32)

//===----------------------------------------------------------------------===//
// Linear expressions
//===----------------------------------------------------------------------===//

__qpu__ void linear_expr0() {
  cudaq::qubit q;
  // 9 iterations: [(10-1)-(0+1)+1]/1
  for (int i = 0; i + 1 < 10; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr0
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : i32
// CHECK:           cc.loop while ((%[[VAL_5:.*]] = %{{.*}}) -> (i32)) {
// CHECK:             %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %{{.*}} : i32
// CHECK:             %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_5]] : i32)
// CHECK:           } do {
// CHECK:           } {normalized}

__qpu__ void linear_expr1a() {
  cudaq::qubit q;
  // 6 iterations: [(11-1)-0+(2*1)]/(2*1)
  for (int i = 0; 2 * i < 11; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr1a
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_6:.*]] = %
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %
// CHECK:             %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_8]](%[[VAL_6]] : i32)
// CHECK:           } {normalized}

__qpu__ void linear_expr1b() {
  cudaq::qubit q;
  // 5 iterations: [(11-1)-1+(2*1)]/(2*1)
  for (int i = 1; 2 * i < 11; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr1b
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_6:.*]] = %
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_8]](%[[VAL_6]] : i32)
// CHECK:           } {normalized}

__qpu__ void linear_expr2() {
  cudaq::qubit q;
  // 7 iterations: [(21-1)-(0+2)+(3*1)]/(3*1)
  for (int i = 0; 3 * i + 2 < 21; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr2
// CHECK:           %[[VAL_2:.*]] = arith.constant 7 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_7:.*]] = %
// CHECK:             %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:             %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_10]](%[[VAL_7]] : i32)
// CHECK:           } {normalized}

__qpu__ void linear_expr3a() {
  cudaq::qubit q;
  // 6 iterations: [(-19+1)-(-15+2)+(-1*1)]/(-1*1)
  for (int i = 15; -1 * i + 2 > -19; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr3a
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 6 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant -1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_7:.*]] = %
// CHECK:             %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:             %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_10]](%[[VAL_7]] : i32)
// CHECK:           } {normalized}

__qpu__ void linear_expr3b() {
  cudaq::qubit q;
  // 0 iterations: [(-2+1)-(-15+2)+(-1*1)]/(-1*1)
  // initial condition: false (-13 > -2)
  for (int i = 15; -1 * i + 2 > -2; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr3b
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant -1 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_0]] : i32
// CHECK:           } {normalized}

__qpu__ void linear_expr3c() {
  cudaq::qubit q;
  // undefined iterations: [(-2+1)-(-15+2)+(-1*1)]/(-1*1)
  // initial condition: true (-13 < -2)
  for (int i = 15; -1 * i + 2 < -2; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr3c
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant -1 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_6:.*]] = %[[VAL_0]]) -> (i32)) {
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_0]] : i32
// CHECK:           } {normalized}

__qpu__ void linear_expr4() {
  cudaq::qubit q;
  // 5 iterations: [(-18+1)-(-15+2)+(-1*1)]/(-1*1)
  for (int i = 15; 2 - i > -18; i++)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr4
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_6:.*]] = %
// CHECK:             %[[VAL_7:.*]] = arith.subi %[[VAL_3]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_8]](%[[VAL_6]] : i32)
// CHECK:           } {normalized}

__qpu__ void linear_expr5a() {
  cudaq::qubit q;
  // 5 iterations: [(17-1)-(-4+2)+(-4*-1)]/(-4*-1)
  for (int i = 1; 2 - 4 * i < 17; i--)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr5a
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_7:.*]] = %
// CHECK:             %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_4]], %[[VAL_8]] : i32
// CHECK:             %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_10]](%[[VAL_7]] : i32)
// CHECK:           } {normalized}

__qpu__ void linear_expr5b() {
  cudaq::qubit q;
  // 4 iterations: [(17-1)-(-4+2)+(-4*-1)]/(-4*-1)
  for (int i = 0; 2 - 4 * i < 17; i--)
    x(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_linear_expr5b
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           cc.loop while ((%[[VAL_6:.*]] = %
// CHECK:             %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.subi %[[VAL_3]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_9]](%[[VAL_6]] : i32)
// CHECK:           } {normalized}

