// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

// Tier-1 counter: protocol attribution for the resource ops.
//   - produce_resource with FlatSymbolRefAttr -> @distill_15to1
//   - produce_resource with #fabric.spec_only<"cult-d15">
//   - transport with FlatSymbolRefAttr -> @ls_handoff
//   - inject with FlatSymbolRefAttr -> @t_inject
//
// Each protocol lands in per_protocol with the correct `kind` and
// `op_count`. Transports also bump transport_in/out on the regions
// they connect.

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
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<factory>
  }
}

fabric.gadget @distill_15to1(%r: !fabric.resource<T>)
    -> !fabric.resource<T> {
  fabric.return %r : !fabric.resource<T>
}

fabric.gadget @ls_handoff(%r: !fabric.resource<T>)
    -> !fabric.resource<T> {
  fabric.return %r : !fabric.resource<T>
}

fabric.gadget @t_inject(%p: !fabric.patch<@steane>, %r: !fabric.resource<T>)
    -> !fabric.patch<@steane> {
  fabric.return %p : !fabric.patch<@steane>
}

fabric.gadget @prog {entry} on @dev() {
  %p = fabric.alloc {code = @steane, region = @C0} : <@steane>
  %r0 = fabric.produce_resource {region = @F0,
                                 resource = #fabric.resource<T>,
                                 protocol = @distill_15to1}
      : !fabric.resource<T>
  %r1 = fabric.produce_resource {region = @F0,
                                 resource = #fabric.resource<T>,
                                 protocol = #fabric.spec_only<"cult-d15">}
      : !fabric.resource<T>
  %r0b = fabric.transport %r0 from @F0 to @C0 {protocol = @ls_handoff}
      : !fabric.resource<T> -> !fabric.resource<T>
  %p1 = fabric.inject %p, %r0b {protocol = @t_inject}
      : (!fabric.patch<@steane>, !fabric.resource<T>) -> !fabric.patch<@steane>
  fabric.discard_resource %r1 : !fabric.resource<T>
  fabric.dealloc %p1 : <@steane>
  fabric.return
}

// CHECK:      fabric.counts =
// CHECK-SAME:   per_protocol = {
// CHECK-SAME:     "cult-d15" = {kind = "production", op_count = 1 : i64}
// CHECK-SAME:     distill_15to1 = {kind = "production", op_count = 1 : i64}
// CHECK-SAME:     ls_handoff = {kind = "transport", op_count = 1 : i64}
// CHECK-SAME:     t_inject = {kind = "injection", op_count = 1 : i64}
// CHECK-SAME:   per_region = {
// CHECK-SAME:     C0 = {
// CHECK-SAME:       inject_count = 1 : i64
// CHECK-SAME:       role = "compute"
// CHECK-SAME:       transport_in = 1 : i64
// CHECK-SAME:       transport_out = 0 : i64
// CHECK-SAME:     F0 = {
// CHECK-SAME:       role = "factory"
// CHECK-SAME:       transport_in = 0 : i64
// CHECK-SAME:       transport_out = 1 : i64
