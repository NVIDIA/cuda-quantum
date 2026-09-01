// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s

// Open patch boundaries receive deterministic carrier layouts before the
// entry body is walked. Operations on a valid input owner must not disappear.

fabric.code @tiny {
  distance = 1 : i64,
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @tiny,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

// CHECK: H 0
// CHECK-NEXT: M 0
fabric.gadget @entry {entry} on @dev(%p: !fabric.patch<@tiny>)
    -> tensor<1xi1> {
  %p0 = fabric.h %p data : !fabric.patch<@tiny>
  %p1, %bits = fabric.mz %p0 data
      : !fabric.patch<@tiny> -> tensor<1xi1>
  fabric.dealloc %p1 : !fabric.patch<@tiny>
  fabric.return %bits : tensor<1xi1>
}
