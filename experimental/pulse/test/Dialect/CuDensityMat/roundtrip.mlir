// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_cudm_basic
func.func @test_cudm_basic() {
  // CHECK: cudm.init_handle
  %h = cudm.init_handle : !cudm.handle

  // CHECK: cudm.create_state
  %s = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f64>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state

  // CHECK: cudm.create_workspace
  %ws = cudm.create_workspace %h : (!cudm.handle) -> !cudm.workspace

  // CHECK: cudm.destroy_state
  cudm.destroy_state %s : !cudm.state
  // CHECK: cudm.destroy_workspace
  cudm.destroy_workspace %ws : !cudm.workspace
  // CHECK: cudm.destroy_handle
  cudm.destroy_handle %h : !cudm.handle
  return
}
