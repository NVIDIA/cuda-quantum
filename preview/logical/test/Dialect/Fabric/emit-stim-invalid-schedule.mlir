// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: not qlx-translate --fabric-to-stim %s 2>&1 | FileCheck %s

// A scheduled interaction must never disappear from emitted Stim. The
// translator verifies that the linked code provides the requested CSS checks
// and fails closed before producing a different circuit.

fabric.code @tiny {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 1 : i64, sz = 0 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @tiny,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

// CHECK: error: 'fabric.cx' op schedule 'hx' requires nonempty hx checks on code @tiny
fabric.gadget @entry {entry} on @dev () -> tensor<3xi1> {
  %p = fabric.alloc {code = @tiny, region = @C0} : !fabric.patch<@tiny>
  %p1 = fabric.cx %p sx -> data {schedule = "hx"} : !fabric.patch<@tiny>
  %p2, %bits = fabric.mz %p1 data
      : !fabric.patch<@tiny> -> tensor<3xi1>
  fabric.dealloc %p2 : !fabric.patch<@tiny>
  fabric.return %bits : tensor<3xi1>
}
