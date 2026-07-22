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

  // CHECK: llvm.call @cudm_evolve_step
  %result = cudm.evolve %h, %op, %s_in, %s_out, %ws {integrator = #cudm<integrator magnus_cf4>, t_start = 0.0 : f64, t_end = 10.0 : f64, num_steps = 10 : i64} : !cudm.handle, !cudm.operator, !cudm.state, !cudm.state, !cudm.workspace -> !cudm.state

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
