// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-count{root=entry result=counts})' | FileCheck %s --check-prefix=COUNT

// A referenced fabric.circuit is the gadget's executable realization, not an
// empty signature stub. The translator must follow that verified edge.

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

fabric.circuit @helper_body() -> tensor<1xi1> {
  %p = fabric.alloc {code = @tiny, region = @C0} : !fabric.patch<@tiny>
  %p0 = fabric.prep_z %p : !fabric.patch<@tiny>
  %p1, %bits = fabric.mz %p0 data
      : !fabric.patch<@tiny> -> tensor<1xi1>
  fabric.dealloc %p1 : !fabric.patch<@tiny>
  fabric.return %bits : tensor<1xi1>
}

fabric.gadget @helper() -> tensor<1xi1> realization @helper_body {
  realization_kind = "circuit"
}

fabric.circuit @body() -> tensor<1xi1> {
  %bits = fabric.call @helper() : () -> tensor<1xi1>
  fabric.return %bits : tensor<1xi1>
}

// CHECK: R 0
// CHECK-NEXT: M 0
// COUNT: fabric.counts = {
// COUNT-SAME: logical_qubits_peak = 1 : i64
// COUNT-SAME: operation_counts = {alloc = 1 : i64, call = 1 : i64, dealloc = 1 : i64, mz = 1 : i64, prep_z = 1 : i64}
fabric.gadget @entry {entry} on @dev() -> tensor<1xi1> realization @body {
  realization_kind = "circuit"
}
