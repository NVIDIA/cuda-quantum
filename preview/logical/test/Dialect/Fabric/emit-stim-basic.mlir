// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s

// Smallest end-to-end Fabric -> Stim translation. A proven trivial one-carrier
// code with no syndrome/gauge structure: prep |0>, Hadamard, measure Z,
// and return the raw measurement record.
//
// Checks: the gate and measurement show up in order with correct qubit
// indices (data starts at 0).

fabric.code @tiny {
  distance = 1 : i64,
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64},
  n = 1 : i64,
  k = 1 : i64,
  r = 0 : i64
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @tiny,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

// CHECK: R 0
// CHECK-NEXT: RX 0
// CHECK-NEXT: H 0
// CHECK-NEXT: M 0
fabric.gadget @entry {entry} on @dev () -> tensor<1xi1> {
  %p = fabric.alloc {code = @tiny, region = @C0} : !fabric.patch<@tiny>
  %p0 = fabric.prep_z %p : !fabric.patch<@tiny>
  %px = fabric.prep_x %p0 : !fabric.patch<@tiny>
  %p1 = fabric.h %px data : !fabric.patch<@tiny>
  %p2, %bits = fabric.mz %p1 data
      : !fabric.patch<@tiny> -> tensor<1xi1>
  fabric.dealloc %p2 : !fabric.patch<@tiny>
  fabric.return %bits : tensor<1xi1>
}
