// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

// fabric-count treats dynamic feedback as a static upper bound: both
// branches contribute to counts, and the yielded patch remains attributable
// to the original region for downstream ops.

fabric.code @tiny {
  distance = 1 : i64,
  partitions = {data = 2 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @tiny,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

fabric.gadget @entry {entry} on @dev() -> tensor<1xi1> {
  %p = fabric.alloc {code = @tiny, region = @C0} : !fabric.patch<@tiny>
  %p0 = fabric.prep_z %p : !fabric.patch<@tiny>
  %p1, %bits = fabric.mz %p0 data [0]
      : !fabric.patch<@tiny> -> tensor<1xi1>
  %cond = arith.constant true
  %p2 = fabric.if %cond -> !fabric.patch<@tiny> {
    %pt = fabric.h %p1 data : !fabric.patch<@tiny>
    fabric.yield %pt : !fabric.patch<@tiny>
  } else {
    %pf = fabric.z %p1 data : !fabric.patch<@tiny>
    fabric.yield %pf : !fabric.patch<@tiny>
  }
  fabric.dealloc %p2 : !fabric.patch<@tiny>
  fabric.return %bits : tensor<1xi1>
}

// CHECK:      fabric.counts =
// CHECK-SAME:   logical_qubits_peak = 1 : i64
// CHECK-SAME:   per_region = {
// CHECK-SAME:     C0 = {
// CHECK-SAME:       gate_counts = {dealloc = 1 : i64, h = 1 : i64, mz = 1 : i64, prep_z = 1 : i64, z = 1 : i64}
