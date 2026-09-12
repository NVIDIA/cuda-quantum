// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s

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

fabric.protocol @apply_h : (!fabric.patch<@tiny>) -> !fabric.patch<@tiny> {
^bb0(%patch: !fabric.patch<@tiny>):
  %next = fabric.h %patch data : !fabric.patch<@tiny>
  fabric.protocol_return %next : !fabric.patch<@tiny>
}

// CHECK: R 0
// CHECK-NEXT: H 0
// CHECK-NEXT: M 0
fabric.gadget @entry {entry} on @dev() -> tensor<1xi1> {
  %p = fabric.alloc {code = @tiny, region = @C0} : !fabric.patch<@tiny>
  %p0 = fabric.prep_z %p : !fabric.patch<@tiny>
  %p1 = fabric.call @apply_h(%p0)
      : (!fabric.patch<@tiny>) -> !fabric.patch<@tiny>
  %p2, %bits = fabric.mz %p1 data
      : !fabric.patch<@tiny> -> tensor<1xi1>
  fabric.dealloc %p2 : !fabric.patch<@tiny>
  fabric.return %bits : tensor<1xi1>
}
