// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --cudm-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @main
func.func @main() {
  // CHECK: llvm.call @cudm_init
  %h = cudm.init_handle : !cudm.handle

  // CHECK: llvm.call @cudm_state_alloc
  %s_in = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %s_out = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state

  // CHECK: llvm.call @cudm_workspace_create
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace

  // CHECK: llvm.call @cudm_operator_create
  %op = cudm.create_operator %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.operator

  // CHECK: llvm.call @cudm_evolve
  // CHECK: llvm.call @cudm_state_capture
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator rk4>, t_start = 0.0 : f64, t_end = 10.0 : f64, num_steps = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state

  // CHECK: llvm.call @cudm_operator_destroy
  cudm.destroy_operator %op : !cudm.operator
  // CHECK: llvm.call @cudm_workspace_destroy
  cudm.destroy_workspace %ws : !cudm.workspace
  // CHECK: llvm.call @cudm_state_destroy
  cudm.destroy_state %s_out : !cudm.state
  cudm.destroy_state %s_in : !cudm.state
  // CHECK: llvm.call @cudm_destroy
  cudm.destroy_handle %h : !cudm.handle
  return
}

// CHECK-LABEL: func @main_magnus
func.func @main_magnus() {
  %h = cudm.init_handle : !cudm.handle
  %s_in = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %s_out = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace
  %op = cudm.create_operator %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.operator
  // The magnus integrator lowers to the dialect enum value 5.
  // CHECK: llvm.mlir.constant(5 : i32)
  // CHECK: llvm.call @cudm_evolve
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator magnus>, t_start = 0.0 : f64, t_end = 10.0 : f64, num_steps = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state
  cudm.destroy_operator %op : !cudm.operator
  cudm.destroy_workspace %ws : !cudm.workspace
  cudm.destroy_state %s_out : !cudm.state
  cudm.destroy_state %s_in : !cudm.state
  cudm.destroy_handle %h : !cudm.handle
  return
}

// CHECK-LABEL: func @main_crank_nicolson
func.func @main_crank_nicolson() {
  %h = cudm.init_handle : !cudm.handle
  %s_in = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %s_out = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace
  %op = cudm.create_operator %h {mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.operator
  // The crank_nicolson integrator lowers to the dialect enum value 6.
  // CHECK: llvm.mlir.constant(6 : i32)
  // CHECK: llvm.call @cudm_evolve
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator crank_nicolson>, t_start = 0.0 : f64, t_end = 10.0 : f64, num_steps = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state
  cudm.destroy_operator %op : !cudm.operator
  cudm.destroy_workspace %ws : !cudm.workspace
  cudm.destroy_state %s_out : !cudm.state
  cudm.destroy_state %s_in : !cudm.state
  cudm.destroy_handle %h : !cudm.handle
  return
}
