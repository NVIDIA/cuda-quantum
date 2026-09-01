// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// Test all Fabric attributes round-trip through parsing and printing.
// Attributes are tested as function argument attributes.

// CHECK-LABEL: func.func @test_partition_attrs
func.func @test_partition_attrs()
    attributes {
      // CHECK-DAG: p_data = #fabric.partition<data>
      p_data = #fabric.partition<data>,
      // CHECK-DAG: p_sx = #fabric.partition<sx>
      p_sx = #fabric.partition<sx>,
      // CHECK-DAG: p_sz = #fabric.partition<sz>
      p_sz = #fabric.partition<sz>,
      // CHECK-DAG: p_all = #fabric.partition<all>
      p_all = #fabric.partition<all>
    } {
  return
}

// CHECK-LABEL: func.func @test_merge_basis_attrs
func.func @test_merge_basis_attrs()
    attributes {
      // CHECK-DAG: mb_x = #fabric.merge_basis<X>
      mb_x = #fabric.merge_basis<X>,
      // CHECK-DAG: mb_z = #fabric.merge_basis<Z>
      mb_z = #fabric.merge_basis<Z>
    } {
  return
}

// CHECK-LABEL: func.func @test_boundary_attrs
func.func @test_boundary_attrs()
    attributes {
      // CHECK-DAG: b_n = #fabric.boundary<north>
      b_n = #fabric.boundary<north>,
      // CHECK-DAG: b_e = #fabric.boundary<east>
      b_e = #fabric.boundary<east>,
      // CHECK-DAG: b_s = #fabric.boundary<south>
      b_s = #fabric.boundary<south>,
      // CHECK-DAG: b_w = #fabric.boundary<west>
      b_w = #fabric.boundary<west>
    } {
  return
}

// CHECK-LABEL: func.func @test_prep_attrs
func.func @test_prep_attrs()
    attributes {
      // CHECK-DAG: prep_z = #fabric.prep<z>
      prep_z = #fabric.prep<z>,
      // CHECK-DAG: prep_x = #fabric.prep<x>
      prep_x = #fabric.prep<x>
    } {
  return
}

// CHECK-LABEL: func.func @test_layout_attrs
func.func @test_layout_attrs()
    attributes {
      // CHECK-DAG: l_cb = #fabric.layout<checkerboard>
      l_cb = #fabric.layout<checkerboard>,
      // CHECK-DAG: l_lin = #fabric.layout<linear>
      l_lin = #fabric.layout<linear>,
      // CHECK-DAG: l_dir = #fabric.layout<direct>
      l_dir = #fabric.layout<direct>,
      // CHECK-DAG: l_cust = #fabric.layout<custom>
      l_cust = #fabric.layout<custom>
    } {
  return
}

// CHECK-LABEL: func.func @test_role_attrs
func.func @test_role_attrs()
    attributes {
      // CHECK-DAG: r_c = #fabric.role<compute>
      r_c = #fabric.role<compute>,
      // CHECK-DAG: r_m = #fabric.role<memory>
      r_m = #fabric.role<memory>,
      // CHECK-DAG: r_f = #fabric.role<factory>
      r_f = #fabric.role<factory>,
      // CHECK-DAG: r_s = #fabric.role<scratch>
      r_s = #fabric.role<scratch>
    } {
  return
}

// CHECK-LABEL: func.func @test_route_attrs
func.func @test_route_attrs()
    attributes {
      // CHECK-DAG: rt_pd = #fabric.route<patch_deform>
      rt_pd = #fabric.route<patch_deform>,
      // CHECK-DAG: rt_auto = #fabric.route<automorphism>
      rt_auto = #fabric.route<automorphism>,
      // CHECK-DAG: rt_swap = #fabric.route<swap>
      rt_swap = #fabric.route<swap>
    } {
  return
}

// CHECK-LABEL: func.func @test_floorplan_attrs
func.func @test_floorplan_attrs()
    attributes {
      // CHECK-DAG: fp1 = #fabric.floorplan<checkerboard, [11, 11]>
      fp1 = #fabric.floorplan<checkerboard, [11, 11]>,
      // CHECK-DAG: fp2 = #fabric.floorplan<direct, [6]>
      fp2 = #fabric.floorplan<direct, [6]>,
      // CHECK-DAG: fp3 = #fabric.floorplan<linear, [3, 5, 2]>
      fp3 = #fabric.floorplan<linear, [3, 5, 2]>
    } {
  return
}

// CHECK-LABEL: func.func @test_flow_attrs
func.func @test_flow_attrs()
    attributes {
      // CHECK-DAG: flow_id = #fabric.flow<{x = "x", z = "z"}>
      flow_id = #fabric.flow<{x = "x", z = "z"}>,
      // CHECK-DAG: flow_h = #fabric.flow<{x = "z", z = "x"}>
      flow_h = #fabric.flow<{x = "z", z = "x"}>,
      // CHECK-DAG: flow_s = #fabric.flow<{x = "y", z = "z"}>
      flow_s = #fabric.flow<{x = "y", z = "z"}>
    } {
  return
}
