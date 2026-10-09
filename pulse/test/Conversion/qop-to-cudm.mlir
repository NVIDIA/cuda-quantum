// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --qop-to-cudm %s | FileCheck %s

module @qop_to_cudm_test attributes {qop.n_qubits = 1 : i64, qop.t_start = 0.0 : f64, qop.t_end = 100.0 : f64, qop.num_steps = 100 : i64} {

// CHECK-LABEL: func @main
func.func @main() {
  %c0 = arith.constant 0 : i64

  // Static Z term
  %sz = qop.spin(%c0) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %coeff_z = qop.const_scalar {real = 15.707963 : f64, imag = 0.0 : f64} : !qop.scalar
  %term_z = qop.make_product(%coeff_z, %sz) : !qop.product

  // Drive X term
  %sx = qop.spin(%c0) {kind = #qop<handler_kind spin_x>} : !qop.handler
  %cb_x = qop.callback_scalar @drive_envelope_0_x : !qop.scalar
  %term_x = qop.make_product(%cb_x, %sx) : !qop.product

  // Non-Hermitian amplitude-damping collapse operator.
  %lowering = qop.spin(%c0) {kind = #qop<handler_kind spin_lowering>} : !qop.handler
  %collapse_coeff = qop.const_scalar {real = 0.1 : f64, imag = 0.0 : f64} : !qop.scalar
  %collapse_product = qop.make_product(%collapse_coeff, %lowering) : !qop.product
  %collapse = qop.make_sum(%collapse_product) : !qop.op

  %H = qop.make_sum(%term_z, %term_x) : !qop.op
  %L = qop.lindblad(%H, %collapse) : !qop.super_op

  // CHECK: cudm.init_handle
  // CHECK: cudm.create_state {{.*}}purity = #cudm<purity mixed>
  // CHECK: cudm.create_workspace
  // CHECK: cudm.create_operator
  // CHECK: %[[Z:[0-9]+]] = cudm.create_elementary_op
  // CHECK: %[[X:[0-9]+]] = cudm.create_elementary_op
  // CHECK: %[[LOWERING:[0-9]+]] = cudm.create_elementary_op
  // CHECK: %[[RAISING:[0-9]+]] = cudm.create_elementary_op
  // CHECK: cudm.create_op_term
  // CHECK: cudm.append_elementary_product
  // CHECK: cudm.append_elementary_product {{.*}} %[[LOWERING]], %[[RAISING]] {{.*}}duality = array<i32: 0, 1>
  // CHECK: cudm.append_elementary_product {{.*}} %[[LOWERING]], %[[RAISING]] {{.*}}duality = array<i32: 0, 0>
  // CHECK: cudm.operator_append_term
  // CHECK: cudm.evolve
  // CHECK: cudm.destroy_operator
  // CHECK: cudm.destroy_workspace
  // CHECK: cudm.destroy_state
  // CHECK: cudm.destroy_handle
  return
}

}
