/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                         {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

CUDAQ_REGISTER_OPERATION(custom_x, 1, 0, {0, 1, 1, 0})

CUDAQ_REGISTER_OPERATION(custom_cnot, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0})

CUDAQ_REGISTER_OPERATION(custom_swap, 2, 0,
                         {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1})

CUDAQ_REGISTER_OPERATION(custom_s, 1, 0,
                         {1, 0, 0, std::complex<double>{0.0, 1.0}})

CUDAQ_REGISTER_OPERATION(toffoli, 3, 0,
                         {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0})

__qpu__ void kernel_1() {
  cudaq::qubit q, r;
  custom_h(q);
  custom_cnot(q, r);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_1._Z8kernel_1v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_h_generator_1._Z20custom_h_generator_{{.*}}vectorId{{.*}} %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_cnot_generator_2._Z23custom_cnot_generator_{{.*}}vectorId{{.*}} %[[VAL_0]], %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_2() {
  cudaq::qubit q, r;
  custom_h(q);
  custom_x<cudaq::ctrl>(q, r);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_2._Z8kernel_2v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_h_generator_1._Z20custom_h_generator_{{.*}}vectorId{{.*}} %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_x_generator_1._Z20custom_x_generator_{{.*}}vectorId{{.*}} {{\[}}%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_3() {
  cudaq::qubit q, r;
  x(q);
  custom_swap(q, r);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_3._Z8kernel_3v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_swap_generator_2._Z23custom_swap_generator_{{.*}}vectorId{{.*}} %[[VAL_0]], %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_4() {
  cudaq::qvector q(4);
  x(q.front(3));
  custom_swap<cudaq::ctrl>(q[0], q[1], q[2], q[3]);
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_4._Z8kernel_4v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_5:.*]] = quake.subveq %[[VAL_4]], 0, 2 : (!quake.veq<4>) -> !quake.veq<3>
// CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_3]]) -> (i64)) {
// CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_0]] : i64
// CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
// CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_9]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:             quake.x %[[VAL_10]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_9]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_2]] : i64
// CHECK:             cc.continue %[[VAL_12]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[VAL_14:.*]] = quake.extract_ref %[[VAL_4]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[VAL_15:.*]] = quake.extract_ref %[[VAL_4]][2] : (!quake.veq<4>) -> !quake.ref
// CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_4]][3] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_swap_generator_2._Z23custom_swap_generator_{{.*}}vectorId{{.*}} {{\[}}%[[VAL_13]], %[[VAL_14]]] %[[VAL_15]], %[[VAL_16]] : (!quake.ref, !quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_5() {
  auto q = cudaq::qubit();
  h(q);
  custom_s(q);
  custom_s<cudaq::adj>(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_5._Z8kernel_5v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_s_generator_1._Z20custom_s_generator_{{.*}}vectorId{{.*}} %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_custom_s_generator_1._Z20custom_s_generator_{{.*}}vectorId{{.*}}<adj> %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_6() {
  cudaq::qvector q(3);
  x(q);
  toffoli(q[0], q[1], q[2]);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_6._Z8kernel_6v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 3 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_1]]) -> (i64)) {
// CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_2]] : i64
// CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
// CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:             quake.x %[[VAL_8]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_7]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_0]] : i64
// CHECK:             cc.continue %[[VAL_10]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_3]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:           quake.custom_op @__nvqpp__mlirgen__function_toffoli_generator_3._Z19toffoli_generator_{{.*}}vectorId{{.*}} %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : (!quake.ref, !quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }
