// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --split-input-file %s --verify-diagnostics | qlx-opt --split-input-file --verify-diagnostics | FileCheck %s

// Positive: qLDPC CodeOp with k, lx, lz round-trips.

// CHECK: fabric.code @toric_3
// CHECK-SAME: k = 2
// CHECK-SAME: lx = [array<i64: 0, 1, 2>, array<i64: 9, 12, 15>]
// CHECK-SAME: lz = [array<i64: 0, 3, 6>, array<i64: 9, 10, 11>]
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

// -----

// Legacy k=1 Steane: no `k`, no `lx`, no `lz` — still round-trips.
// CHECK: fabric.code @steane_legacy
// CHECK-NOT: k =
// CHECK-NOT: lx =
// CHECK-NOT: lz =
fabric.code @steane_legacy {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

// -----

// Negative: len(lx) must equal k.
// expected-error@below {{lx must have exactly k = 2 entries, got 1}}
fabric.code @bad_k_mismatch {
  distance = 3 : i64,
  partitions = {data = 18 : i64, sx = 9 : i64, sz = 9 : i64},
  k = 2 : i64,
  lx = [array<i64: 0, 1, 2>]
}

// -----

// Negative: logical-op qubit index must be within data partition.
// expected-error@below {{lz[0][2] qubit index 99 out of range for data partition size 18}}
fabric.code @bad_oob_index {
  distance = 3 : i64,
  partitions = {data = 18 : i64, sx = 9 : i64, sz = 9 : i64},
  k = 2 : i64,
  lz = [array<i64: 0, 3, 99>, array<i64: 9, 10, 11>]
}

// -----

// Negative: k must be nonnegative; k=0 is a native stabilizer-state code.
// expected-error@below {{k must be >= 0, got -1}}
fabric.code @bad_negative_k {
  distance = 3 : i64,
  partitions = {data = 18 : i64, sx = 9 : i64, sz = 9 : i64},
  k = -1 : i64
}

// (An "empty logical-op row" case is not reachable through the MLIR
// textual parser — `array<i64: >` is a parse error before the verifier
// runs.  The verifier still guards against empty inner arrays when the
// op is built programmatically from C++/Python, exercised by the
// Python test `test_empty_logical_op_rejected`.)
