// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s

// A fabric.code carrying only the CSS support-array adapter form (no
// canonical symplectic attributes) must still satisfy the CSS algebra
// over GF(2).

// X and Z checks on the same odd-size support anticommute.
// expected-error @+1 {{css checks violate Hx.Hz^T = 0 at 0,0}}
fabric.code @anticommuting_checks {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 1 : i64, sz = 1 : i64},
  hx = [array<i64: 0, 1, 2>],
  hz = [array<i64: 0, 1, 2>]
}

// -----

// lz here is a stabilizer element (hz[1] + hz[2]), so it overlaps the
// logical X representative evenly instead of pairing with it.
// expected-error @+1 {{css checks violate Lx.Lz^T = I at 0,0}}
fabric.code @unpaired_logicals {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64},
  hx = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  hz = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  lx = [array<i64: 4, 5, 6>],
  lz = [array<i64: 1, 2, 5, 6>]
}

// -----

// Declared n/k imply s = 4 independent stabilizers, but the Steane checks
// only carry one logical qubit's worth (rank 6 = n - 1).
// expected-error @+1 {{css checks have GF(2) stabilizer rank 6 but n/k/r imply 4 independent stabilizers}}
fabric.code @wrong_rank {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64},
  n = 7 : i64,
  k = 3 : i64,
  hx = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  hz = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>]
}

// -----

// Check supports must stay inside [0, n).
// expected-error @+1 {{hx[0][1] qubit index 7 out of range for 7 data qubit(s)}}
fabric.code @out_of_range_support {
  distance = 1 : i64,
  partitions = {data = 7 : i64, sx = 1 : i64, sz = 0 : i64},
  n = 7 : i64,
  hx = [array<i64: 0, 7>]
}
