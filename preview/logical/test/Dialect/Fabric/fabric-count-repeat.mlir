// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

// Tier-1 counter: `fabric.repeat` multiplies gate counts inside its body
// by `count`. One Hadamard inside `repeat 3` → 3 total. Nested repeats
// compose multiplicatively.

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
  %p = fabric.alloc {code = @steane, region = @C0} : !fabric.patch<@steane>
  %p_out = fabric.repeat 3 iter(%pi : !fabric.patch<@steane> = %p) {
    %p1 = fabric.h %pi data : !fabric.patch<@steane>
    fabric.yield %p1 : !fabric.patch<@steane>
  }
  fabric.dealloc %p_out : !fabric.patch<@steane>
  fabric.return
}

// CHECK:      fabric.counts =
// CHECK-SAME:   gate_counts = {dealloc = 1 : i64, h = 3 : i64}
