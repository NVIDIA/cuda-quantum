// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// Round-trip of the renamed produce_resource / discard_resource ops and
// the !fabric.resource<R> type. Covers T, CCZ, and CS, with both
// symbol-ref and spec_only protocol attributes.

fabric.code @c15 {
  distance = 15 : i64,
  partitions = {data = 225 : i64, sx = 112 : i64, sz = 112 : i64}
}

fabric.gadget @distill_15to1(%r: !fabric.resource<T>)
    -> !fabric.resource<T> {
  fabric.return %r : !fabric.resource<T>
}

fabric.gadget @produce_three_T() {
  // CHECK: fabric.produce_resource
  // CHECK-SAME: protocol = @distill_15to1
  // CHECK-SAME: resource = #fabric.resource<T>
  %r0 = fabric.produce_resource {region = @F0,
                                 resource = #fabric.resource<T>,
                                 protocol = @distill_15to1}
      : !fabric.resource<T>
  // CHECK: fabric.produce_resource
  // CHECK-SAME: protocol = #fabric.spec_only<"distill-15to1-T">
  %r1 = fabric.produce_resource {region = @F1,
                                 resource = #fabric.resource<T>,
                                 protocol = #fabric.spec_only<"distill-15to1-T">}
      : !fabric.resource<T>
  // CHECK: fabric.produce_resource
  // CHECK-SAME: resource = #fabric.resource<CCZ>
  %r2 = fabric.produce_resource {region = @F2,
                                 resource = #fabric.resource<CCZ>}
      : !fabric.resource<CCZ>
  // CHECK: fabric.discard_resource
  fabric.discard_resource %r0 : !fabric.resource<T>
  // CHECK: fabric.discard_resource
  fabric.discard_resource %r1 : !fabric.resource<T>
  // CHECK: fabric.discard_resource
  fabric.discard_resource %r2 : !fabric.resource<CCZ>
  fabric.return
}
