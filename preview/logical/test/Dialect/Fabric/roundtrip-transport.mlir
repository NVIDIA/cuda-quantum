// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// Verifies fabric.transport round-trips with both a symbol-ref protocol
// (the compiled gadget case) and a #fabric.spec_only<> protocol (the
// estimation-only case).

fabric.code @surface_15 {
  distance = 15 : i64,
  partitions = {data = 225 : i64, sx = 112 : i64, sz = 112 : i64}
}

fabric.gadget @ls_handoff(%r: !fabric.resource<T>) -> !fabric.resource<T> {
  fabric.return %r : !fabric.resource<T>
}

fabric.gadget @roundtrip_test(%r: !fabric.resource<T>)
    -> !fabric.resource<T> {
  // CHECK: fabric.transport
  // CHECK-SAME: from @F0 to @C0
  // CHECK-SAME: protocol = @ls_handoff
  %r2 = fabric.transport %r from @F0 to @C0 {protocol = @ls_handoff}
      : !fabric.resource<T> -> !fabric.resource<T>
  // CHECK: fabric.transport
  // CHECK-SAME: protocol = #fabric.spec_only<"ls-handoff-spec">
  %r3 = fabric.transport %r2 from @C0 to @M0
      {protocol = #fabric.spec_only<"ls-handoff-spec">}
      : !fabric.resource<T> -> !fabric.resource<T>
  fabric.return %r3 : !fabric.resource<T>
}
