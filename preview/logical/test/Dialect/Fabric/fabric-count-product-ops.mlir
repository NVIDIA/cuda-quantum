// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --fabric-count %s | FileCheck %s

fabric.code @toric_3 {
  distance = 3 : i64,
  partitions = {data = 18 : i64, sx = 8 : i64, sz = 8 : i64},
  n = 18 : i64,
  k = 2 : i64,
  r = 0 : i64,
  hx = [array<i64: 0, 1, 2, 3, 4, 5>, array<i64: 3, 4, 5, 6, 7, 8>, array<i64: 9, 10>, array<i64: 10, 11>, array<i64: 12, 13>, array<i64: 13, 14>, array<i64: 15, 16>, array<i64: 16, 17>],
  hz = [array<i64: 0, 1>, array<i64: 1, 2>, array<i64: 3, 4>, array<i64: 4, 5>, array<i64: 6, 7>, array<i64: 7, 8>, array<i64: 9, 10, 11, 12, 13, 14>, array<i64: 12, 13, 14, 15, 16, 17>],
  lx = [array<i64: 0, 1, 2>, array<i64: 9, 12, 15>],
  lz = [array<i64: 0, 3, 6>, array<i64: 9, 10, 11>]
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @toric_3,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @prog {entry} on @dev() -> i1 {
  %p = fabric.alloc {code = @toric_3, region = @C0} : <@toric_3>
  %p1, %m = fabric.measure_product %p {
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "ZZ"
  } : (!fabric.patch<@toric_3>) -> (!fabric.patch<@toric_3>, i1)
  %p2 = fabric.rotate_product %p1 {
    angle = 2.500000e-01 : f64,
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "ZX",
    synthesis = "auto"
  } : (!fabric.patch<@toric_3>) -> !fabric.patch<@toric_3>
  fabric.selection %m {mode = "abort_on", accept_when = false} : i1
  fabric.dealloc %p2 : <@toric_3>
  fabric.return %m : i1
}

// CHECK: gate_counts =
// CHECK-SAME: dealloc = 1 : i64
// CHECK-SAME: measure_product = 1 : i64
// CHECK-SAME: rotate_product = 1 : i64
// CHECK: success_count = 1 : i64
