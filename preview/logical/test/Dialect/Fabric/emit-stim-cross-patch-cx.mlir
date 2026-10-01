// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: qlx-translate --fabric-to-stim %s | FileCheck %s

// Cross-patch fabric.cx: a two-patch CX between physical qubits living
// in two different patches (different codes/regions). Exercises
// EmitStim::emitCrossTwoQubit, which resolves the `pairs="c:t,..."`
// string into absolute Stim qubit indices via each patch's PatchInfo.
//
// Qubit numbering (allocation order): patch @A is allocated first, so
// its data partition is [0, 2); patch @B is next, data partition [2, 4).
// `pairs = "index"` therefore maps (ctrl A.data[0]=0 -> targ B.data[0]=2)
// and (ctrl A.data[1]=1 -> targ B.data[1]=3): `CX 0 2 1 3`.
//
// The point of the primitive: it only needs each patch's .data/.sx/.sz
// partitions — it does not care which code each patch runs.

fabric.code @a {
  distance = 1 : i64,
  partitions = {data = 2 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.code @b {
  distance = 1 : i64,
  partitions = {data = 2 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @dev {
  fabric.region @A {
    code = @a,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<checkerboard, [1, 2]>
  }
  fabric.region @B {
    code = @b,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<checkerboard, [1, 2]>
  }
}

// CHECK: R 0 1
// CHECK-NEXT: R 2 3
// CHECK-NEXT: CX 0 2 1 3
// CHECK: M 0 1
// CHECK: M 2 3
fabric.gadget @entry {entry} on @dev () -> (tensor<2xi1>, tensor<2xi1>) {
  %pa = fabric.alloc {code = @a, region = @A} : !fabric.patch<@a>
  %pb = fabric.alloc {code = @b, region = @B} : !fabric.patch<@b>
  %pa0 = fabric.reset %pa data : !fabric.patch<@a>
  %pb0 = fabric.reset %pb data : !fabric.patch<@b>
  // Cross-patch CX: ctrl partition on patches[0]=@A, targ on patches[1]=@B.
  %pa1, %pb1 = fabric.cx %pa0, %pb0 data -> data {pairs = "index"}
      : (!fabric.patch<@a>, !fabric.patch<@b>) -> (!fabric.patch<@a>, !fabric.patch<@b>)
  %pa2, %abits = fabric.mz %pa1 data
      : !fabric.patch<@a> -> tensor<2xi1>
  %pb2, %bbits = fabric.mz %pb1 data
      : !fabric.patch<@b> -> tensor<2xi1>
  fabric.dealloc %pa2 : !fabric.patch<@a>
  fabric.dealloc %pb2 : !fabric.patch<@b>
  fabric.return %abits, %bbits : tensor<2xi1>, tensor<2xi1>
}
