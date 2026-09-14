// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

// Tier-1 counter: a cross-region transversal_cx between a compute
// region and a factory region. Each region's gate_counts records the
// transversal_cx op once; transversal_edges records the [C0, F0] pair.

fabric.code @steane {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @steane,
    floorplan = #fabric.floorplan<direct, [4]>,
    role = #fabric.role<compute>
  }
  fabric.region @F0 {
    code = @steane,
    floorplan = #fabric.floorplan<direct, [4]>,
    role = #fabric.role<factory>
  }
  fabric.interconnect @compute_to_factory {
    region_a = @C0, port_a = 0 : i64,
    region_b = @F0, port_b = 0 : i64
  }
}

fabric.gadget @prog {entry} on @dev() {
  %pc = fabric.alloc {code = @steane, region = @C0} : <@steane>
  %pf = fabric.alloc {code = @steane, region = @F0} : <@steane>
  %pc_out, %pf_out = fabric.transversal_cx %pc, %pf
      : (!fabric.patch<@steane>, !fabric.patch<@steane>)
       -> (!fabric.patch<@steane>, !fabric.patch<@steane>)
  fabric.dealloc %pc_out : <@steane>
  fabric.dealloc %pf_out : <@steane>
  fabric.return
}

// CHECK:      fabric.counts =
// CHECK-SAME:   logical_qubits_peak = 2 : i64
// CHECK-SAME:   per_region = {
// CHECK-SAME:     C0 = {
// CHECK-SAME:       gate_counts = {dealloc = 1 : i64, transversal_cx = 1 : i64}
// CHECK-SAME:       role = "compute"
// CHECK-SAME:     F0 = {
// CHECK-SAME:       gate_counts = {dealloc = 1 : i64, transversal_cx = 1 : i64}
// CHECK-SAME:       role = "factory"
// CHECK-SAME:   transversal_edges = {{\[\[}}"C0", "F0"]]
