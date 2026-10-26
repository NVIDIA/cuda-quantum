// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --fabric-count | FileCheck %s

// The child needs two live reservations to avoid a saturating release. That
// requirement must propagate into the enclosing summary. Splitting a patch
// changes SSA ownership but does not increment FabricCount's live count: the
// second call must walk rather than replay the summary built at count two.
fabric.code @one {distance = 1 : i64, partitions = {data = 1 : i64}}
fabric.machine @dev {
  fabric.region @C0 {
    code = @one, floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}
fabric.gadget @replace(%a: !fabric.patch<@one>, %b: !fabric.patch<@one>)
    -> (!fabric.patch<@one>, !fabric.patch<@one>) {
  fabric.dealloc %a : !fabric.patch<@one>
  fabric.dealloc %b : !fabric.patch<@one>
  %a0 = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  %b0 = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.return %a0, %b0 : !fabric.patch<@one>, !fabric.patch<@one>
}
fabric.gadget @outer(%a: !fabric.patch<@one>, %b: !fabric.patch<@one>)
    -> (!fabric.patch<@one>, !fabric.patch<@one>) {
  %a0, %b0 = fabric.call @replace(%a, %b)
      : (!fabric.patch<@one>, !fabric.patch<@one>)
        -> (!fabric.patch<@one>, !fabric.patch<@one>)
  fabric.return %a0, %b0 : !fabric.patch<@one>, !fabric.patch<@one>
}
fabric.gadget @entry {entry} on @dev() {
  %a = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  %b = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  %a0, %b0 = fabric.call @outer(%a, %b)
      : (!fabric.patch<@one>, !fabric.patch<@one>)
        -> (!fabric.patch<@one>, !fabric.patch<@one>)
  fabric.dealloc %a0 : !fabric.patch<@one>
  fabric.dealloc %b0 : !fabric.patch<@one>
  %c = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  %c0, %c1, %bit = fabric.split %c
      : (!fabric.patch<@one>) -> (!fabric.patch<@one>, !fabric.patch<@one>, i1)
  %d0, %d1 = fabric.call @outer(%c0, %c1)
      : (!fabric.patch<@one>, !fabric.patch<@one>)
        -> (!fabric.patch<@one>, !fabric.patch<@one>)
  %extra = fabric.alloc {code = @one, region = @C0} : !fabric.patch<@one>
  fabric.dealloc %d0 : !fabric.patch<@one>
  fabric.dealloc %d1 : !fabric.patch<@one>
  fabric.dealloc %extra : !fabric.patch<@one>
  fabric.return
}
// CHECK: fabric.counts = {
// CHECK-SAME: gadget_calls = {outer = 2 : i64, replace = 2 : i64}
// CHECK-SAME: logical_qubits_peak = 3 : i64
// CHECK-SAME: patches_peak = 3 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: patches = 3 : i64
