// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

fabric.code @toric_3 {
  distance = 3 : i64,
  partitions = {data = 18 : i64, sx = 9 : i64, sz = 9 : i64},
  k = 2 : i64,
  lx = [array<i64: 0, 1, 2>, array<i64: 9, 12, 15>],
  lz = [array<i64: 0, 3, 6>, array<i64: 9, 10, 11>]
}

fabric.gadget @products(%p: !fabric.patch<@toric_3>) -> (!fabric.patch<@toric_3>, i1) {
  %p1, %m = fabric.measure_product %p {
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "ZX"
  } : (!fabric.patch<@toric_3>) -> (!fabric.patch<@toric_3>, i1)
  %p2 = fabric.rotate_product %p1 {
    angle = 5.000000e-01 : f64,
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "ZZ",
    synthesis = "auto"
  } : (!fabric.patch<@toric_3>) -> !fabric.patch<@toric_3>
  fabric.return %p2, %m : !fabric.patch<@toric_3>, i1
}

// CHECK-LABEL: fabric.gadget @products
// CHECK: fabric.measure_product
// CHECK-SAME: logical_indices = array<i64: 0, 1>
// CHECK-SAME: patch_indices = array<i64: 0, 0>
// CHECK-SAME: pauli_product = "ZX"
// CHECK: fabric.rotate_product
// CHECK-SAME: angle = 5.000000e-01 : f64
// CHECK-SAME: pauli_product = "ZZ"
// CHECK-SAME: synthesis = "auto"

// A single optional leading '-' on the product string encodes a negated
// product (sign -1). It must survive the round trip on every product op,
// including the physical-view fabric.mpp form.
fabric.gadget @negated_products(%p: !fabric.patch<@toric_3>) -> (!fabric.patch<@toric_3>, i1) {
  %p1, %m = fabric.measure_product %p {
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "-ZX"
  } : (!fabric.patch<@toric_3>) -> (!fabric.patch<@toric_3>, i1)
  %p2 = fabric.rotate_product %p1 {
    angle = 5.000000e-01 : f64,
    logical_indices = array<i64: 0, 1>,
    patch_indices = array<i64: 0, 0>,
    pauli_product = "-ZZ",
    synthesis = "auto"
  } : (!fabric.patch<@toric_3>) -> !fabric.patch<@toric_3>
  %p3, %bits = fabric.mpp %p2 data indices [0, 1] paulis "-XX" {record = "signed"}
      : !fabric.patch<@toric_3> -> tensor<1xi1>
  fabric.return %p3, %m : !fabric.patch<@toric_3>, i1
}

// CHECK-LABEL: fabric.gadget @negated_products
// CHECK: fabric.measure_product
// CHECK-SAME: pauli_product = "-ZX"
// CHECK: fabric.rotate_product
// CHECK-SAME: pauli_product = "-ZZ"
// CHECK: fabric.mpp
// CHECK-SAME: paulis "-XX"

fabric.code @subsystem {
  distance = 1 : i64,
  partitions = {data = 2 : i64},
  k = 1 : i64,
  r = 1 : i64
}

fabric.gadget @gauge_product(%p: !fabric.patch<@subsystem>)
    -> (!fabric.patch<@subsystem>, i1) {
  %next, %outcome = fabric.measure_product %p {
    logical_indices = array<i64: 1>,
    patch_indices = array<i64: 0>,
    pauli_product = "X",
    subsystem_indices = array<i64: 0>,
    subsystem_kinds = ["gauge"]
  } : (!fabric.patch<@subsystem>) -> (!fabric.patch<@subsystem>, i1)
  fabric.return %next, %outcome : !fabric.patch<@subsystem>, i1
}

// CHECK-LABEL: fabric.gadget @gauge_product
// CHECK: fabric.measure_product
// CHECK-SAME: logical_indices = array<i64: 1>
// CHECK-SAME: subsystem_indices = array<i64: 0>
// CHECK-SAME: subsystem_kinds = ["gauge"]
