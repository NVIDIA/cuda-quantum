// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s --fabric-count 2>&1 | FileCheck %s

// A newly live caller region needs a different tick summary; changing only
// its patch count must reuse it. Freeing that region restores the first key.
fabric.code @one {distance = 1 : i64, partitions = {data = 1 : i64}}
fabric.machine @dev {
  fabric.region @C0 {
    code = @one, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
  fabric.region @C1 {
    code = @one, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}
fabric.gadget @tick() {
  fabric.tick
  fabric.return
}
fabric.gadget @entry {entry} on @dev() {
  %p = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.call @tick() : () -> ()
  %b0 = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  fabric.call @tick() : () -> ()
  %b1 = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  fabric.call @tick() : () -> ()
  fabric.dealloc %b0 : !fabric.patch<@one>
  fabric.dealloc %b1 : !fabric.patch<@one>
  fabric.call @tick() : () -> ()
  fabric.dealloc %p : !fabric.patch<@one>
  fabric.return
}
// CHECK: callable-summary-probes=2 callable-summary-hits=4
// CHECK: fabric.counts = {
// CHECK-SAME: logical_qubits_peak = 3 : i64
// CHECK-SAME: patches_peak = 3 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: patches = 1 : i64
// CHECK-SAME: rounds_by_kind = {tick = 4 : i64}
// CHECK-SAME: C1 = {
// CHECK-SAME: patches = 2 : i64
// CHECK-SAME: rounds_by_kind = {tick = 2 : i64}
