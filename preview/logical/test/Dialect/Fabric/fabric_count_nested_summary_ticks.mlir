// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s --fabric-count 2>&1 | FileCheck %s

// C1 belongs to the enclosing callable. A nested summary must observe its
// temporary release; only the tick after replacement counts for C1. The
// inherited caller context C0 advances on both ticks. Repeat the child and
// the enclosing callable to exercise both cache hits and bound composition.
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
fabric.gadget @replace(%p: !fabric.patch<@one>) -> !fabric.patch<@one> {
  fabric.dealloc %p : !fabric.patch<@one>
  fabric.tick
  %out = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  fabric.tick
  fabric.return %out : !fabric.patch<@one>
}
fabric.gadget @outer() {
  %p = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  %a = fabric.call @replace(%p) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  %b = fabric.call @replace(%a) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  fabric.dealloc %b : !fabric.patch<@one>
  fabric.return
}
fabric.gadget @entry {entry} on @dev() {
  %p = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.call @outer() : () -> ()
  %b = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.call @outer() : () -> ()
  fabric.dealloc %b : !fabric.patch<@one>
  fabric.dealloc %p : !fabric.patch<@one>
  fabric.return
}
// CHECK: callable-summary-probes=2 callable-summary-hits=4
// CHECK: fabric.counts = {
// CHECK-SAME: gadget_calls = {outer = 2 : i64, replace = 4 : i64}
// CHECK-SAME: logical_qubits_peak = 3 : i64
// CHECK-SAME: patches_peak = 3 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: patches = 2 : i64
// CHECK-SAME: rounds_by_kind = {tick = 8 : i64}
// CHECK-SAME: C1 = {
// CHECK-SAME: patches = 1 : i64
// CHECK-SAME: rounds_by_kind = {tick = 4 : i64}
