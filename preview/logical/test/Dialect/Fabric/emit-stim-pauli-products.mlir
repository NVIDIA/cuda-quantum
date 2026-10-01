// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s

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
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

// CHECK: R 0 1 2 3
// CHECK-NEXT: MPP !X2*Y3
// CHECK-NEXT: MPP Y0*X1*Z2
// CHECK-NEXT: MPP !Y0*Y1
fabric.gadget @entry {entry} on @dev() -> (tensor<1xi1>, i1, i1) {
  %p = fabric.alloc {code = @two_logicals, region = @C0}
      : !fabric.patch<@two_logicals>
  %p0 = fabric.reset %p all : !fabric.patch<@two_logicals>
  %p1, %physical = fabric.mpp %p0 data indices [2, 3] paulis "-XY"
      : !fabric.patch<@two_logicals> -> tensor<1xi1>
  %p2, %logical_y = fabric.measure_product %p1 {
    logical_indices = array<i64: 0>,
    patch_indices = array<i64: 0>,
    pauli_product = "Y"
  } : (!fabric.patch<@two_logicals>)
      -> (!fabric.patch<@two_logicals>, i1)
  %p3, %overlap = fabric.measure_product %p2 {
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "XZ"
  } : (!fabric.patch<@two_logicals>)
      -> (!fabric.patch<@two_logicals>, i1)
  fabric.dealloc %p3 : !fabric.patch<@two_logicals>
  fabric.return %physical, %logical_y, %overlap : tensor<1xi1>, i1, i1
}
