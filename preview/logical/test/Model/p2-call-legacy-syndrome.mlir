// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

// Alpha QLX threaded a syndrome beside each patch at call sites while its
// patch-only gate gadgets omitted those values from their signatures. This is
// accepted narrowly as import compatibility; canonical QLX calls use the
// exact callee signature.
module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.gadget @h(%patch: !fabric.patch<@c>) -> !fabric.patch<@c> {
    fabric.return %patch : !fabric.patch<@c>
  }
  fabric.gadget @entry(
      %patch: !fabric.patch<@c>,
      %syndrome: !fabric.syndrome<@c>)
      -> (!fabric.patch<@c>, !fabric.syndrome<@c>) {
    %0:2 = fabric.call @h(%patch, %syndrome)
      : (!fabric.patch<@c>, !fabric.syndrome<@c>)
      -> (!fabric.patch<@c>, !fabric.syndrome<@c>)
    fabric.return %0#0, %0#1
      : !fabric.patch<@c>, !fabric.syndrome<@c>
  }
}

// CHECK: fabric.call @h(%arg0, %arg1)
// CHECK-SAME: (!fabric.patch<@c>, !fabric.syndrome<@c>)
