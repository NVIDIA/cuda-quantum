// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

fabric.code @prepared_zero {
  distance = 0 : i64,
  hz = [array<i64: 0>],
  k = 0 : i64,
  lx = [],
  lz = [],
  n = 1 : i64,
  partitions = {data = 1 : i64},
  r = 0 : i64
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @prepared_zero,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @entry {entry} on @dev() {
  %state = fabric.alloc {code = @prepared_zero, region = @C0}
      : !fabric.patch<@prepared_zero>
  fabric.dealloc %state : !fabric.patch<@prepared_zero>
  fabric.return
}

// CHECK: fabric.counts =
// CHECK-SAME: logical_qubits_peak = 0 : i64
// CHECK-SAME: patches_peak = 1 : i64
// CHECK-SAME: per_region = {C0 = {
// CHECK-SAME: patches = 1 : i64
