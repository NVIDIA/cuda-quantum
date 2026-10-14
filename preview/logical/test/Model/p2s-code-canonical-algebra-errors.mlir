// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s -split-input-file 2>&1 | FileCheck %s

// The canonical symplectic basis is verified algebraically: replacing the
// second kept stabilizer Z1Z2 with X1X2 anticommutes with Z0Z1 at one site.
module {
  // CHECK: stabilizers violates canonical commutation at 0,1
  fabric.code @noncommuting_repetition {anti_stabilizers = dense<[[false, true, true, false, false, false], [false, false, true, false, false, false]]> : tensor<2x6xi1>, distance = 3 : i64, encoding_clifford = dense<[[false, true, true, false, false, false], [false, false, true, false, false, false], [true, true, true, false, false, false], [false, false, false, true, true, false], [false, false, false, false, true, true], [false, false, false, true, false, false]]> : tensor<6x6xi1>, gauge_x_basis = dense<> : tensor<0x6xi1>, gauge_z_basis = dense<> : tensor<0x6xi1>, hz = [array<i64: 0, 1>, array<i64: 1, 2>], k = 1 : i64, logical_x_basis = dense<[[true, true, true, false, false, false]]> : tensor<1x6xi1>, logical_z_basis = dense<[[false, false, false, true, false, false]]> : tensor<1x6xi1>, lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>], metadata = {distance_status = "claimed"}, n = 3 : i64, partitions = {data = 3 : i64, sx = 0 : i64, sz = 2 : i64}, r = 0 : i64, stabilizer_basis = dense<[[false, false, false, true, true, false], [false, true, true, false, false, false]]> : tensor<2x6xi1>}
}

// -----

// A dependent kept-stabilizer basis cannot reach the declared rank
// s = n - k - r.
module {
  // CHECK: canonical code basis does not have the required rank
  fabric.code @dependent_repetition {anti_stabilizers = dense<[[false, true, true, false, false, false], [false, false, true, false, false, false]]> : tensor<2x6xi1>, distance = 3 : i64, encoding_clifford = dense<[[false, true, true, false, false, false], [false, false, true, false, false, false], [true, true, true, false, false, false], [false, false, false, true, true, false], [false, false, false, false, true, true], [false, false, false, true, false, false]]> : tensor<6x6xi1>, gauge_x_basis = dense<> : tensor<0x6xi1>, gauge_z_basis = dense<> : tensor<0x6xi1>, hz = [array<i64: 0, 1>, array<i64: 1, 2>], k = 1 : i64, logical_x_basis = dense<[[true, true, true, false, false, false]]> : tensor<1x6xi1>, logical_z_basis = dense<[[false, false, false, true, false, false]]> : tensor<1x6xi1>, lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>], metadata = {distance_status = "claimed"}, n = 3 : i64, partitions = {data = 3 : i64, sx = 0 : i64, sz = 2 : i64}, r = 0 : i64, stabilizer_basis = dense<[[false, false, false, true, true, false], [false, false, false, true, true, false]]> : tensor<2x6xi1>}
}
