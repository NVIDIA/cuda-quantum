/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

#include <complex>
using namespace std::complex_literals;

CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                         {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

CUDAQ_REGISTER_OPERATION(custom_x, 1, 0, {0, 1, 1, 0})

CUDAQ_REGISTER_OPERATION(custom_cnot, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0})

CUDAQ_REGISTER_OPERATION(custom_swap, 2, 0,
                         {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1})

CUDAQ_REGISTER_OPERATION(custom_s, 1, 0, {1, 0, 0, 1i})

CUDAQ_REGISTER_OPERATION(
    my_u3_generator_0, 1, 3,
    {std::cos(parameters[0] / 2.),
     -std::exp(i *parameters[2]) * std::sin(parameters[0] / 2.),
     std::exp(i *parameters[1]) * std::sin(parameters[0] / 2.),
     std::exp(i *(parameters[2] + parameters[1])) *
         std::cos(parameters[0] / 2.)})

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
// CHECK:           quake.unitary %[[VAL_0]] : (!quake.ref) -> () {generator = @custom_h_generator_1}
// CHECK:           quake.unitary %[[VAL_0]], %[[VAL_1]] : (!quake.ref, !quake.ref) -> () {generator = @custom_cnot_generator_2}
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
// CHECK:           quake.unitary %[[VAL_0]] : (!quake.ref) -> () {generator = @custom_h_generator_1}
// CHECK:           quake.unitary {{\[}}%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> () {generator = @custom_x_generator_1}
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
// CHECK:           quake.unitary %[[VAL_0]], %[[VAL_1]] : (!quake.ref, !quake.ref) -> () {generator = @custom_swap_generator_2}
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_4() {
  cudaq::qvector q(4);
  x(q.front(3));
  custom_swap<cudaq::ctrl>(q[0], q[1], q[2], q[3]);
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_4._Z8kernel_4v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_5:.*]] = quake.subveq %[[VAL_4]], %[[VAL_3]], %[[VAL_1]] : (!quake.veq<4>, i64, i64) -> !quake.veq<3>
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
// CHECK:           quake.unitary {{\[}}%[[VAL_13]], %[[VAL_14]]] %[[VAL_15]], %[[VAL_16]] : (!quake.ref, !quake.ref, !quake.ref, !quake.ref) -> () {generator = @custom_swap_generator_2}
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
// CHECK:           quake.unitary %[[VAL_0]] : (!quake.ref) -> () {generator = @custom_s_generator_1}
// CHECK:           quake.unitary<adj> %[[VAL_0]] : (!quake.ref) -> () {generator = @custom_s_generator_1}
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_6() {
  cudaq::qubit q;
  my_u3_generator_0(M_PI, M_PI, M_PI_2, q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_6._Z8kernel_6v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 3.1415926535897931 : f64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<f64>
// CHECK:           %[[VAL_4:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_4]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           quake.unitary (%[[VAL_6]], %[[VAL_7]], %[[VAL_8]]) %[[VAL_2]] : (f64, f64, f64, !quake.ref) -> () {generator = @my_u3_generator_0_generator_1}
// CHECK:           return
// CHECK:         }

__qpu__ void kernel_7() {
  cudaq::qvector q(3);
  x(q);
  toffoli(q[0], q[1], q[2]);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel_7._Z8kernel_7v() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : i64
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<3>
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
// CHECK:           quake.unitary %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : (!quake.ref, !quake.ref, !quake.ref) -> () {generator = @toffoli_generator_3}
// CHECK:           return
// CHECK:         }