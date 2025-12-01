/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
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
    	cudaq::qvector r(2);
    	g1();
      	for (int i = 0; i < 10; ++i) {
	 		if (f1(i)) {
	    		cudaq::qubit q;
	    		x<cudaq::ctrl>(q,r[0]);
	    		break;
	 		}
	 		x<cudaq::ctrl>(r[0],r[1]);
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__C()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               %[[VAL_10:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_11:.*]] = func.call @_Z2f1i(%[[VAL_10]]) : (i32) -> i1
// CHECK:               cf.cond_br %[[VAL_11]], ^bb1, ^bb2
// CHECK:             ^bb1:
// CHECK:               %[[VAL_12:.*]] = quake.alloca !quake.ref
// CHECK:               %[[VAL_13:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_12]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               quake.dealloc %[[VAL_12]] : !quake.ref
// CHECK:               cc.break
// CHECK:             ^bb2:
// CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_14]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               func.call @_Z2g2v() : () -> ()
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_17:.*]] = func.call @_Z2f2i(%[[VAL_16]]) : (i32) -> i1
// CHECK:               cf.cond_br %[[VAL_17]], ^bb3, ^bb4
// CHECK:             ^bb3:
// CHECK:               %[[VAL_18:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.y %[[VAL_18]] : (!quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             ^bb4:
// CHECK:               func.call @_Z2g3v() : () -> ()
// CHECK:               %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:                 %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_0]] : i64
// CHECK:                 cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_22:.*]]: i64):
// CHECK:                 %[[VAL_23:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_22]]] : (!quake.veq<2>, i64) -> !quake.ref
// CHECK:                 quake.z %[[VAL_23]] : (!quake.ref) -> ()
// CHECK:                 cc.continue %[[VAL_22]] : i64
// CHECK:               } step {
// CHECK:               ^bb0(%[[VAL_24:.*]]: i64):
// CHECK:                 %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
// CHECK:                 cc.continue %[[VAL_25]] : i64
// CHECK:               } {invariant}
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : i32
// CHECK:               cc.store %[[VAL_27]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_28:.*]] = quake.mz %[[VAL_6]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

struct D {
   	void operator()() __qpu__ {
      	cudaq::qvector r(2);
      	g1();
      	for (int i = 0; i < 10; ++i) {
	 		if (f1(i)) {
	    		cudaq::qubit q;
	    		x<cudaq::ctrl>(q,r[0]);
	    		continue;
	 		}
	 		x<cudaq::ctrl>(r[0],r[1]);
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__D() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               %[[VAL_10:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_11:.*]] = func.call @_Z2f1i(%[[VAL_10]]) : (i32) -> i1
// CHECK:               cf.cond_br %[[VAL_11]], ^bb1, ^bb2
// CHECK:             ^bb1:
// CHECK:               %[[VAL_12:.*]] = quake.alloca !quake.ref
// CHECK:               %[[VAL_13:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_12]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               quake.dealloc %[[VAL_12]] : !quake.ref
// CHECK:               cc.continue
// CHECK:             ^bb2:
// CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.x {{\[}}%[[VAL_14]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               func.call @_Z2g2v() : () -> ()
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_17:.*]] = func.call @_Z2f2i(%[[VAL_16]]) : (i32) -> i1
// CHECK:               cf.cond_br %[[VAL_17]], ^bb3, ^bb4
// CHECK:             ^bb3:
// CHECK:               %[[VAL_18:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.y %[[VAL_18]] : (!quake.ref) -> ()
// CHECK:               cc.break
// CHECK:             ^bb4:
// CHECK:               func.call @_Z2g3v() : () -> ()
// CHECK:               %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:                 %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_0]] : i64
// CHECK:                 cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_22:.*]]: i64):
// CHECK:                 %[[VAL_23:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_22]]] : (!quake.veq<2>, i64) -> !quake.ref
// CHECK:                 quake.z %[[VAL_23]] : (!quake.ref) -> ()
// CHECK:                 cc.continue %[[VAL_22]] : i64
// CHECK:               } step {
// CHECK:               ^bb0(%[[VAL_24:.*]]: i64):
// CHECK:                 %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
// CHECK:                 cc.continue %[[VAL_25]] : i64
// CHECK:               } {invariant}
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : i32
// CHECK:               cc.store %[[VAL_27]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_28:.*]] = quake.mz %[[VAL_6]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

struct E {
   void operator()() __qpu__ {
    	cudaq::qvector r(2);
      	g1();
      	for (int i = 0; i < 10; ++i) {
	 		if (f1(i)) {
	    		cudaq::qubit q;
	    		x<cudaq::ctrl>(q,r[0]);
	    		return;
	 		}
	 		x<cudaq::ctrl>(r[0],r[1]);
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__E() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           %[[VAL_7:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:           cf.cond_br %[[VAL_9]], ^bb2, ^bb7
// CHECK:         ^bb2:
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = call @_Z2f1i(%[[VAL_10]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_11]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           %[[VAL_12:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_12]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           quake.dealloc %[[VAL_12]] : !quake.ref
// CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<2>
// CHECK:           return
// CHECK:         ^bb4:
// CHECK:           %[[VAL_14:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_14]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           call @_Z2g2v() : () -> ()
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = call @_Z2f2i(%[[VAL_16]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_17]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK:           %[[VAL_18:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.y %[[VAL_18]] : (!quake.ref) -> ()
// CHECK:           cf.br ^bb7
// CHECK:         ^bb6:
// CHECK:           call @_Z2g3v() : () -> ()
// CHECK:           %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_22:.*]]: i64):
// CHECK:             %[[VAL_23:.*]] = quake.extract_ref %[[VAL_6]][%[[VAL_22]]] : (!quake.veq<2>, i64) -> !quake.ref
// CHECK:             quake.z %[[VAL_23]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_22]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_25]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : i32
// CHECK:           cc.store %[[VAL_27]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb7:
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_28:.*]] = quake.mz %[[VAL_6]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<2>
// CHECK:           return

struct F {
	void operator()() __qpu__ {
      	cudaq::qvector r(2);
      	g1();
      	for (int i = 0; i < 10; ++i) {
	 		if (f1(i)) {
	    		cudaq::qubit q;
	    		x<cudaq::ctrl>(q,r[0]);
	    		continue;
	 		}
	 		x<cudaq::ctrl>(r[0],r[1]);
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__F() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 10 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
// CHECK:           call @_Z2g1v() : () -> ()
// CHECK:           %[[VAL_7:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:           cf.cond_br %[[VAL_9]], ^bb2, ^bb8
// CHECK:         ^bb2:
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = call @_Z2f1i(%[[VAL_10]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_11]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK:           %[[VAL_12:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_12]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           quake.dealloc %[[VAL_12]] : !quake.ref
// CHECK:           cf.br ^bb7
// CHECK:         ^bb4:
// CHECK:           %[[VAL_14:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.x [%[[VAL_14]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           call @_Z2g2v() : () -> ()
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = call @_Z2f2i(%[[VAL_16]]) : (i32) -> i1
// CHECK:           cf.cond_br %[[VAL_17]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK:           %[[VAL_18:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.y %[[VAL_18]] : (!quake.ref) -> ()
// CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<2>
// CHECK:           return
// CHECK:         ^bb6:
// CHECK:           call @_Z2g3v() : () -> ()
// CHECK:           %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_22:.*]]: i64):
// CHECK:             %[[VAL_23:.*]] = quake.extract_ref %[[VAL_6]][%[[VAL_22]]] : (!quake.veq<2>, i64) -> !quake.ref
// CHECK:             quake.z %[[VAL_23]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_22]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
// CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_25]] : i64
// CHECK:           } {invariant}
// CHECK:           cf.br ^bb7
// CHECK:         ^bb7:
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : i32
// CHECK:           cc.store %[[VAL_27]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           cf.br ^bb1
// CHECK:         ^bb8:
// CHECK:           call @_Z2g4v() : () -> ()
// CHECK:           %[[VAL_28:.*]] = quake.mz %[[VAL_6]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<2>
// CHECK:           return
