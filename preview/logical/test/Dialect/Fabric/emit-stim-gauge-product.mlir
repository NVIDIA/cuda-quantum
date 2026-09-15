// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s

fabric.code @bacon_shor_2x3 {
  distance = 2 : i64,
  partitions = {data = 6 : i64, gauge_ancilla = 2 : i64},
  n = 6 : i64,
  k = 1 : i64,
  r = 2 : i64,
  hx = [array<i64: 0, 1, 2, 3, 4, 5>],
  hz = [array<i64: 0, 1, 3, 4>, array<i64: 1, 2, 4, 5>],
  gx = [array<i64: 1, 4>, array<i64: 2, 5>],
  gz = [array<i64: 0, 1>, array<i64: 0, 2>],
  lx = [array<i64: 0, 1, 2>],
  lz = [array<i64: 0, 3>]
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @bacon_shor_2x3,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

// CHECK: R 0 1 2 3 4 5 6 7
// CHECK-NEXT: MPP X1*X4
fabric.gadget @entry {entry} on @dev() -> i1 {
  %p = fabric.alloc {code = @bacon_shor_2x3, region = @C0}
      : !fabric.patch<@bacon_shor_2x3>
  %p0 = fabric.reset %p all : !fabric.patch<@bacon_shor_2x3>
  %p1, %outcome = fabric.measure_product %p0 {
    logical_indices = array<i64: 1>,
    patch_indices = array<i64: 0>,
    pauli_product = "X",
    subsystem_indices = array<i64: 0>,
    subsystem_kinds = ["gauge"]
  } : (!fabric.patch<@bacon_shor_2x3>)
      -> (!fabric.patch<@bacon_shor_2x3>, i1)
  fabric.dealloc %p1 : !fabric.patch<@bacon_shor_2x3>
  fabric.return %outcome : i1
}
