// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --fabric-count | FileCheck %s

// A middle summary must retain the child's upper bound on C1 liveness.
// Both calls have C1 live at entry, but only the first loses its last C1
// reservation during replacement. Reusing that summary at count two would
// incorrectly omit a tick. C0 remains unconditional inherited tick context.
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
fabric.gadget @middle(%p: !fabric.patch<@one>) -> !fabric.patch<@one> {
  %a = fabric.call @replace(%p) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  fabric.return %a : !fabric.patch<@one>
}
fabric.gadget @one_patch() {
  %p = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  %a = fabric.call @middle(%p) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  fabric.dealloc %a : !fabric.patch<@one>
  fabric.return
}
fabric.gadget @two_patches() {
  %p = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  %background = fabric.alloc {code = @one, region = @C1} : !fabric.patch<@one>
  %a = fabric.call @middle(%p) : (!fabric.patch<@one>) -> !fabric.patch<@one>
  fabric.dealloc %a : !fabric.patch<@one>
  fabric.dealloc %background : !fabric.patch<@one>
  fabric.return
}
fabric.gadget @entry {entry} on @dev() {
  %p = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.call @one_patch() : () -> ()
  fabric.call @two_patches() : () -> ()
  fabric.dealloc %p : !fabric.patch<@one>
  fabric.return
}
// CHECK: fabric.counts = {
// CHECK-SAME: logical_qubits_peak = 3 : i64
// CHECK-SAME: patches_peak = 3 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: rounds_by_kind = {tick = 4 : i64}
// CHECK-SAME: C1 = {
// CHECK-SAME: patches = 2 : i64
// CHECK-SAME: rounds_by_kind = {tick = 3 : i64}
