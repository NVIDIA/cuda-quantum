/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// RUN: cudaq-quake %s | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
// clang-format on

#include "cudaq.h"

struct test {
  cudaq::qview<> q;
  cudaq::qview<> r;
};

__qpu__ void applyH(cudaq::qubit &q) { h(q); }
__qpu__ void applyX(cudaq::qubit &q) { x(q); }
__qpu__ void kernel(test t) {
  h(t.q);
  s(t.r);

  applyH(t.q[0]);
  applyX(t.r[0]);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernel4test(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.struq<!quake.veq<?>, !quake.veq<?>>) attributes {"cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = quake.get_member %[[VAL_0]][0] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_12:.*]] = quake.get_member %[[VAL_0]][1] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_13:.*]] = quake.veq_size %[[VAL_12]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_21:.*]] = quake.get_member %[[VAL_0]][0] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_22:.*]] = quake.extract_ref %[[VAL_21]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__function_applyH._Z6applyHRN5cudaq5quditILm2EEE(%[[VAL_22]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_23:.*]] = quake.get_member %[[VAL_0]][1] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_24:.*]] = quake.extract_ref %[[VAL_23]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__function_applyX._Z6applyXRN5cudaq5quditILm2EEE(%[[VAL_24]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void entry_initlist() {
  cudaq::qvector q(2), r(2);
  test tt{q, r};
  kernel(tt);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry_initlist._Z14entry_initlistv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.make_struq %[[VAL_0]], %[[VAL_1]] : (!quake.veq<2>, !quake.veq<2>) -> !quake.struq<!quake.veq<?>, !quake.veq<?>>
// CHECK:           call @__nvqpp__mlirgen__function_kernel._Z6kernel4test(%[[VAL_2]]) : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void entry_ctor() {
  cudaq::qvector q(2), r(2);
  test tt(q, r);
  h(tt.r[0]);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry_ctor._Z10entry_ctorv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// QIR-LABEL: define void @__nvqpp__mlirgen__function_kernel._Z6kernel4test({ 
// QIR-SAME:    %Array*, %Array* } %[[VAL_1:.*]]) local_unnamed_addr {
// QIR:         %[[VAL_2:.*]] = extractvalue { %Array*, %Array* } %[[VAL_1]], 0
// QIR:         %[[VAL_3:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%Array* %[[VAL_2]])
// QIR:         %[[VAL_4:.*]] = icmp sgt i64 %[[VAL_3]], 0
// QIR:         br i1 %[[VAL_4]], label %[[VAL_5:.*]], label %[[VAL_6:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_7:.*]], %[[VAL_5]]
// QIR:         %[[VAL_8:.*]] = phi i64 [ %[[VAL_9:.*]], %[[VAL_5]] ], [ 0, %[[VAL_7]] ]
// QIR:         %[[VAL_10:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_2]], i64 %[[VAL_8]])
// QIR:         %[[VAL_13:.*]] = load %Qubit*, %Qubit** %[[VAL_10]], align 8
// QIR:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_13]])
// QIR:         %[[VAL_9]] = add nuw nsw i64 %[[VAL_8]], 1
// QIR:         %[[VAL_14:.*]] = icmp eq i64 %[[VAL_9]], %[[VAL_3]]
// QIR:         br i1 %[[VAL_14]], label %[[VAL_6]], label %[[VAL_5]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_5]], %[[VAL_7]]
// QIR:         %[[VAL_15:.*]] = extractvalue { %Array*, %Array* } %[[VAL_1]], 1
// QIR:         %[[VAL_16:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%Array* %[[VAL_15]])
// QIR:         %[[VAL_17:.*]] = icmp sgt i64 %[[VAL_16]], 0
// QIR:         br i1 %[[VAL_17]], label %[[VAL_18:.*]], label %[[VAL_19:.*]]
// QIR:       .lr.ph3:                                          ; preds = %[[VAL_6]], %[[VAL_18]]
// QIR:         %[[VAL_20:.*]] = phi i64 [ %[[VAL_21:.*]], %[[VAL_18]] ], [ 0, %[[VAL_6]] ]
// QIR:         %[[VAL_22:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_15]], i64 %[[VAL_20]])
// QIR:         %[[VAL_24:.*]] = load %Qubit*, %Qubit** %[[VAL_22]]
// QIR:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_24]])
// QIR:         %[[VAL_21]] = add nuw nsw i64 %[[VAL_20]], 1
// QIR:         %[[VAL_25:.*]] = icmp eq i64 %[[VAL_21]], %[[VAL_16]]
// QIR:         br i1 %[[VAL_25]], label %[[VAL_19]], label %[[VAL_18]]
// QIR:       ._crit_edge4:                                     ; preds = %[[VAL_18]], %[[VAL_6]]
// QIR:         %[[VAL_26:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_2]], i64 0)
// QIR:         %[[VAL_28:.*]] = load %Qubit*, %Qubit** %[[VAL_26]]
// QIR:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_28]])
// QIR:         %[[VAL_29:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_15]], i64 0)
// QIR:         %[[VAL_31:.*]] = load %Qubit*, %Qubit** %[[VAL_29]]
// QIR:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_31]])
// QIR:         ret void
// QIR:       }

// QIR-LABEL: define void @__nvqpp__mlirgen__function_entry_initlist._Z14entry_initlistv() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 4)
// QIR:         %[[VAL_2:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 0)
// QIR:         %[[VAL_5:.*]] = load %Qubit*, %Qubit** %[[VAL_2]], align 8
// QIR:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_5]])
// QIR:         %[[VAL_6:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 1)
// QIR:         %[[VAL_8:.*]] = load %Qubit*, %Qubit** %[[VAL_6]]
// QIR:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_8]])
// QIR:         %[[VAL_9:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 2)
// QIR:         %[[VAL_11:.*]] = load %Qubit*, %Qubit** %[[VAL_9]]
// QIR:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_11]])
// QIR:         %[[VAL_12:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 3)
// QIR:         %[[VAL_14:.*]] = load %Qubit*, %Qubit** %[[VAL_12]]
// QIR:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_14]])
// QIR:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_5]])
// QIR:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_11]])
// QIR:         tail call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_0]])
// QIR:         ret void
// QIR:       }

// QIR-LABEL: define void @__nvqpp__mlirgen__function_entry_ctor._Z10entry_ctorv() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 4)
// QIR:         %[[VAL_2:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 2)
// QIR:         %[[VAL_5:.*]] = load %Qubit*, %Qubit** %[[VAL_2]], align 8
// QIR:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_5]])
// QIR:         tail call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_0]])
// QIR:         ret void
// QIR:       }
// clang-format on
