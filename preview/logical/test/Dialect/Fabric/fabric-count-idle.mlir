// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

// Tier-1 counter: `fabric.idle` records the `rounds` attribute as the
// `idle` bucket of `rounds_by_kind` — *not* as a gate, and the value
// is the rounds count, not the op count.

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

fabric.gadget @prog {entry} on @dev() {
  %p = fabric.alloc {code = @steane, region = @C0} : <@steane>
  %p1 = fabric.idle %p {rounds = 5 : i64} : <@steane>
  %p2 = fabric.idle %p1 {rounds = 2 : i64} : <@steane>
  fabric.dealloc %p2 : <@steane>
  fabric.return
}

// CHECK:      fabric.counts =
// CHECK-SAME:   gate_counts = {dealloc = 1 : i64}
// CHECK-SAME:   rounds_by_kind = {idle = 7 : i64}
