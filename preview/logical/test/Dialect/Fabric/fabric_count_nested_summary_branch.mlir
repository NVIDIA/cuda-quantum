// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --fabric-count='root=entry' | FileCheck %s

// A nested summary must retain a region's zero live-count entry after freeing
// its scratch allocation, just as the inline branch does. Otherwise the
// exact branch-ownership comparison sees different maps and rejects the join.
fabric.code @one {distance = 1 : i64, partitions = {data = 1 : i64}}
fabric.machine @dev {
  fabric.region @C0 {
    code = @one, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}
fabric.gadget @scratch() {
  %p = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.dealloc %p : !fabric.patch<@one>
  fabric.return
}
fabric.gadget @outer() {
  %condition = arith.constant true
  cflow.if %condition {
    fabric.call @scratch() : () -> ()
    cflow.yield
  } else {
    %p = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
    fabric.dealloc %p : !fabric.patch<@one>
    cflow.yield
  }
  fabric.return
}
fabric.gadget @entry() {
  fabric.call @outer() : () -> ()
  fabric.call @outer() : () -> ()
  fabric.return
}
// CHECK: fabric.counts = {
// CHECK-SAME: gadget_calls = {outer = 2 : i64, scratch = 2 : i64}
// CHECK-SAME: logical_qubits_peak = 1 : i64
// CHECK-SAME: patches_peak = 1 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: gate_counts = {dealloc = 4 : i64}
// CHECK-SAME: patches = 1 : i64
