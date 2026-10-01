// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

// Tier-1 static counter: smallest end-to-end program.
// Single Steane region, three Hadamards, one CX, one Z-measurement.

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
}

fabric.gadget @prog {entry} on @dev() -> tensor<7xi1> {
  %p = fabric.alloc {code = @steane, region = @C0} : !fabric.patch<@steane>
  %p1 = fabric.h %p data : !fabric.patch<@steane>
  %p2 = fabric.h %p1 data : !fabric.patch<@steane>
  %p3 = fabric.h %p2 data : !fabric.patch<@steane>
  %after_cx = fabric.cx %p3 data -> data {schedule = "hx"} : <@steane>
  %pout, %bits = fabric.mz %after_cx data
    : !fabric.patch<@steane> -> tensor<7xi1>
  fabric.dealloc %pout : !fabric.patch<@steane>
  fabric.return %bits : tensor<7xi1>
}

// CHECK:      fabric.counts =
// CHECK-SAME:   logical_qubits_peak = 1 : i64
// CHECK-SAME:   per_protocol = {}
// CHECK-SAME:   per_region = {
// CHECK-SAME:     C0 = {
// CHECK-SAME:       code = "steane"
// CHECK-SAME:       distance = 3 : i64
// CHECK-SAME:       gate_counts = {cx = 1 : i64, dealloc = 1 : i64, h = 3 : i64, mz = 1 : i64}
// CHECK-SAME:       patches = 1 : i64
// CHECK-SAME:       role = "compute"
// CHECK-SAME:   transversal_edges = []
