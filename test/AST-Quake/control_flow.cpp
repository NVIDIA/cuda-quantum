/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --unwind-lowering --canonicalize | FileCheck %s

#include <cudaq.h>

bool f1(int);
bool f2(int);

void g1();
void g2();
void g3();
void g4();

struct C {
   void operator()() __qpu__ {
      cudaq::qreg r(2);
      g1();
      for (int i = 0; i < 10; ++i) {
	 if (f1(i)) {
	    cudaq::qubit q;
	    x(q,r[0]);
	    break;
	 }
	 x(r[0],r[1]);
	 g2();
	 if (f2(i)) {
	    y(r[1]);
	    continue;
	 }
	 g3();
	 z(r);
      }
      g4();
      mz(r);
   }
};

struct D {
   void operator()() __qpu__ {
      cudaq::qreg r(2);
      g1();
      for (int i = 0; i < 10; ++i) {
	 if (f1(i)) {
	    cudaq::qubit q;
	    x(q,r[0]);
	    continue;
	 }
	 x(r[0],r[1]);
	 g2();
	 if (f2(i)) {
	    y(r[1]);
	    break;
	 }
	 g3();
	 z(r);
      }
      g4();
      mz(r);
   }
};

struct E {
   void operator()() __qpu__ {
      cudaq::qreg r(2);
      g1();
      for (int i = 0; i < 10; ++i) {
	 if (f1(i)) {
	    cudaq::qubit q;
	    x(q,r[0]);
	    return;
	 }
	 x(r[0],r[1]);
	 g2();
	 if (f2(i)) {
	    y(r[1]);
	    break;
	 }
	 g3();
	 z(r);
      }
      g4();
      mz(r);
   }
};

struct F {
   void operator()() __qpu__ {
      cudaq::qreg r(2);
      g1();
      for (int i = 0; i < 10; ++i) {
	 if (f1(i)) {
	    cudaq::qubit q;
	    x(q,r[0]);
	    continue;
	 }
	 x(r[0],r[1]);
	 g2();
	 if (f2(i)) {
	    y(r[1]);
	    return;
	 }
	 g3();
	 z(r);
      }
      g4();
      mz(r);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__C()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_9:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_7]], %[[VAL_9]][] : memref<i32>
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:             cf.cond_br %[[VAL_11]], ^bb2, ^bb8
// CHECK:           ^bb2:
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_13:.*]] = func.call @_Z2f1i(%[[VAL_12]]) : (i32) -> i1
// CHECK:             cf.cond_br %[[VAL_13]], ^bb3, ^bb4
// CHECK:           ^bb3:
// CHECK:             %[[VAL_14:.*]] = quake.alloca !quake.ref
// CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x [%[[VAL_14]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
// CHECK:             quake.dealloc %[[VAL_14]] : !quake.ref
// CHECK:             cf.br ^bb8
// CHECK:           ^bb4:
// CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x [%[[VAL_16]]] %[[VAL_17]] : (!quake.ref,
// CHECK:             func.call @_Z2g2v() : () -> ()
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_19:.*]] = func.call @_Z2f2i(%[[VAL_18]]) : (i32) -> i1
// CHECK:             cf.cond_br %[[VAL_19]], ^bb5, ^bb6
// CHECK:           ^bb5:
// CHECK:             %[[VAL_20:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.y %[[VAL_20]] :
// CHECK:             cf.br ^bb7
// CHECK:           ^bb6:
// CHECK:             func.call @_Z2g3v() : () -> ()
// CHECK:             %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_4]]) -> (index)) {
// CHECK:               %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_0]] : index
// CHECK:               cc.condition %[[VAL_23]](%[[VAL_22]] : index)
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_24:.*]]: index):
// CHECK:               %[[VAL_25:.*]] = quake.extract_ref %[[VAL_8]][%[[VAL_24]]] : (!quake.veq<2>, index) -> !quake.ref
// CHECK:               quake.z %[[VAL_25]] : (!quake.ref) -> ()
// CHECK:               cc.continue %[[VAL_24]] : index
// CHECK:             } step {
// CHECK:             ^bb0(%[[VAL_26:.*]]: index):
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : index
// CHECK:               cc.continue %[[VAL_27]] : index
// CHECK:             } {counted}
// CHECK:             cf.br ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_28:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_5]] : i32
// CHECK:             memref.store %[[VAL_29]], %[[VAL_9]][] : memref<i32>
// CHECK:             cf.br ^bb1
// CHECK:           ^bb8:
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_30:.*]] = quake.mz %[[VAL_8]] : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__D()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_9:.*]] = memref.alloca() : memref<i32>
// CHECK:             memref.store %[[VAL_7]], %[[VAL_9]][] : memref<i32>
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:             cf.cond_br %[[VAL_11]], ^bb2, ^bb8
// CHECK:           ^bb2:
// CHECK:             %[[VAL_12:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_13:.*]] = func.call @_Z2f1i(%[[VAL_12]]) : (i32) -> i1
// CHECK:             cf.cond_br %[[VAL_13]], ^bb3, ^bb4
// CHECK:           ^bb3:
// CHECK:             %[[VAL_14:.*]] = quake.alloca !quake.ref
// CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x [%[[VAL_14]]] %[[VAL_15]] :
// CHECK:             quake.dealloc %[[VAL_14]] : !quake.ref
// CHECK:             cf.br ^bb7
// CHECK:           ^bb4:
// CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x [%[[VAL_16]]] %[[VAL_17]] : (!quake.ref, !quake.ref) -> ()
// CHECK:             func.call @_Z2g2v() : () -> ()
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_19:.*]] = func.call @_Z2f2i(%[[VAL_18]]) : (i32) -> i1
// CHECK:             cf.cond_br %[[VAL_19]], ^bb5, ^bb6
// CHECK:           ^bb5:
// CHECK:             %[[VAL_20:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.y %[[VAL_20]]
// CHECK:             cf.br ^bb8
// CHECK:           ^bb6:
// CHECK:             func.call @_Z2g3v() : () -> ()
// CHECK:             %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_4]]) -> (index)) {
// CHECK:               %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_0]] : index
// CHECK:               cc.condition %[[VAL_23]](%[[VAL_22]] : index)
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_24:.*]]: index):
// CHECK:               %[[VAL_25:.*]] = quake.extract_ref %[[VAL_8]][%[[VAL_24]]] : (!quake.veq<2>, index) -> !quake.ref
// CHECK:               quake.z %[[VAL_25]] : (!quake.ref) -> ()
// CHECK:               cc.continue %[[VAL_24]] : index
// CHECK:             } step {
// CHECK:             ^bb0(%[[VAL_26:.*]]: index):
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : index
// CHECK:               cc.continue %[[VAL_27]] : index
// CHECK:             } {counted}
// CHECK:             cf.br ^bb7
// CHECK:           ^bb7:
// CHECK:             %[[VAL_28:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_5]] : i32
// CHECK:             memref.store %[[VAL_29]], %[[VAL_9]][] : memref<i32>
// CHECK:             cf.br ^bb1
// CHECK:           ^bb8:
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_30:.*]] = quake.mz %[[VAL_8]] : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__E()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           %[[VAL_9:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_9]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:           cf.cond_br %[[VAL_11]], ^bb2, ^bb7
// CHECK:         ^bb2:
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_13:.*]] = call @_Z2f1i(%[[VAL_12]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_13]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           %[[VAL_14:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_14]]] %[[VAL_15]] : (!quake.ref, !quake.ref)
// CHECK:           quake.dealloc %[[VAL_14]] : !quake.ref
// CHECK:           cf.br ^bb8
// CHECK:         ^bb4:
// CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_17:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_16]]] %[[VAL_17]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           call @_Z2g2v() : () -> ()
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_19:.*]] = call @_Z2f2i(%[[VAL_18]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_19]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK:           %[[VAL_20:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.y %[[VAL_20]]
// CHECK:           cf.br ^bb7
// CHECK:         ^bb6:
// CHECK:           call @_Z2g3v() : () -> ()
// CHECK:           %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_4]]) -> (index)) {
// CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_0]] : index
// CHECK:             cc.condition %[[VAL_23]](%[[VAL_22]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_24:.*]]: index):
// CHECK:             %[[VAL_25:.*]] = quake.extract_ref %[[VAL_8]][%[[VAL_24]]] : (!quake.veq<2>, index) -> !quake.ref
// CHECK:             quake.z %[[VAL_25]]
// CHECK:             cc.continue %[[VAL_24]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_26:.*]]: index):
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : index
// CHECK:             cc.continue %[[VAL_27]] : index
// CHECK:           } {counted}
// CHECK:           %[[VAL_28:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_5]] : i32
// CHECK:           memref.store %[[VAL_29]], %[[VAL_9]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb7:
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_30:.*]] = quake.mz %[[VAL_8]] : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           cf.br ^bb8
// CHECK:         ^bb8:
// CHECK:           quake.dealloc %[[VAL_8]] : !quake.veq<2>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__F()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           %[[VAL_9:.*]] = memref.alloca() : memref<i32>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_9]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:           cf.cond_br %[[VAL_11]], ^bb2, ^bb8
// CHECK:         ^bb2:
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_13:.*]] = call @_Z2f1i(%[[VAL_12]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_13]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           %[[VAL_14:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_14]]] %[[VAL_15]]
// CHECK:           quake.dealloc %[[VAL_14]] : !quake.ref
// CHECK:           cf.br ^bb7
// CHECK:         ^bb4:
// CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_17:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_16]]] %[[VAL_17]]
// CHECK:           call @_Z2g2v() : () -> ()
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_19:.*]] = call @_Z2f2i(%[[VAL_18]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_19]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK:           %[[VAL_20:.*]] = quake.extract_ref %[[VAL_8]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.y %[[VAL_20]] : (!quake.ref)
// CHECK:           cf.br ^bb9
// CHECK:         ^bb6:
// CHECK:           call @_Z2g3v() : () -> ()
// CHECK:           %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_4]]) -> (index)) {
// CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_0]] : index
// CHECK:             cc.condition %[[VAL_23]](%[[VAL_22]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_24:.*]]: index):
// CHECK:             %[[VAL_25:.*]] = quake.extract_ref %[[VAL_8]][%[[VAL_24]]] : (!quake.veq<2>, index) -> !quake.ref
// CHECK:             quake.z %[[VAL_25]]
// CHECK:             cc.continue %[[VAL_24]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_26:.*]]: index):
// CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : index
// CHECK:             cc.continue %[[VAL_27]] : index
// CHECK:           } {counted}
// CHECK:           cf.br ^bb7
// CHECK:         ^bb7:
// CHECK:           %[[VAL_28:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_5]] : i32
// CHECK:           memref.store %[[VAL_29]], %[[VAL_9]][] : memref<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb8:
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_30:.*]] = quake.mz %[[VAL_8]] : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           cf.br ^bb9
// CHECK:         ^bb9:
// CHECK:           quake.dealloc %[[VAL_8]] : !quake.veq<2>
// CHECK:           return
// CHECK:         }

