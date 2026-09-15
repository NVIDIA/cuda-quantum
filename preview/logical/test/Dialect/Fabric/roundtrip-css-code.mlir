// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

// CSS support-array adapter form only (no canonical symplectic attributes):
// the verifier must accept a genuine CSS algebra and round-trip it.

// Steane [[7,1,3]]: self-dual checks, weight-3 logical representatives.
// Every hx/hz row pair overlaps evenly; lx/lz overlap each check evenly and
// each other oddly; rank(hx) + rank(hz) = 6 = n - k.
// CHECK: fabric.code @steane_css
// CHECK-SAME: k = 1 : i64
// CHECK-SAME: n = 7 : i64
fabric.code @steane_css {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64},
  n = 7 : i64,
  k = 1 : i64,
  hx = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  hz = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  lx = [array<i64: 4, 5, 6>],
  lz = [array<i64: 4, 5, 6>]
}

// 2x3 Bacon-Shor subsystem code (qubit (i,j) = 3i+j): s = 3 = n - k - r,
// with r = 2 canonical gauge pairs. Gauge X operators are vertical pairs,
// gauge Z operators live on row 0; each gx[i]/gz[j] pair overlaps oddly
// exactly on the diagonal and evenly against the stabilizers and logicals.
// CHECK: fabric.code @bacon_shor_2x3
// CHECK-SAME: r = 2 : i64
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
