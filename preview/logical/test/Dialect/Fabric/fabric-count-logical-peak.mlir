// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

fabric.code @two_logicals {
  distance = 2 : i64,
  partitions = {data = 4 : i64, sx = 0 : i64, sz = 0 : i64},
  n = 4 : i64,
  k = 2 : i64,
  r = 0 : i64,
  hx = [array<i64: 0, 1, 2, 3>],
  hz = [array<i64: 0, 1, 2, 3>],
  lx = [array<i64: 0, 1>, array<i64: 0, 2>],
  lz = [array<i64: 0, 2>, array<i64: 0, 1>]
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @two_logicals,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @entry {entry} on @dev() {
  %patch = fabric.alloc {code = @two_logicals, region = @C0}
      : !fabric.patch<@two_logicals>
  fabric.dealloc %patch : !fabric.patch<@two_logicals>
  fabric.return
}

// CHECK: fabric.counts =
// CHECK-SAME: logical_qubits_peak = 2 : i64
// CHECK-SAME: patches_peak = 1 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: patches = 1 : i64
