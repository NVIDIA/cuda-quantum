// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --emit-bytecode %s -o %t && qlx-opt %t | FileCheck %s

// The canonical code algebra round-trips through MLIR bytecode with every
// verified basis intact.
module attributes {qlx.ir_version = "0.4-draft"} {
  // CHECK: fabric.code @bytecode_repetition
  // CHECK-SAME: stabilizer_basis = dense<{{\[\[}}false, false, false, true, true, false], [false, false, false, false, true, true]]>
  fabric.code @bytecode_repetition {anti_stabilizers = dense<[[false, true, true, false, false, false], [false, false, true, false, false, false]]> : tensor<2x6xi1>, distance = 3 : i64, encoding_clifford = dense<[[false, true, true, false, false, false], [false, false, true, false, false, false], [true, true, true, false, false, false], [false, false, false, true, true, false], [false, false, false, false, true, true], [false, false, false, true, false, false]]> : tensor<6x6xi1>, gauge_x_basis = dense<> : tensor<0x6xi1>, gauge_z_basis = dense<> : tensor<0x6xi1>, hz = [array<i64: 0, 1>, array<i64: 1, 2>], k = 1 : i64, logical_x_basis = dense<[[true, true, true, false, false, false]]> : tensor<1x6xi1>, logical_z_basis = dense<[[false, false, false, true, false, false]]> : tensor<1x6xi1>, lx = [array<i64: 0, 1, 2>], lz = [array<i64: 0>], metadata = {distance_status = "claimed"}, n = 3 : i64, partitions = {data = 3 : i64, sx = 0 : i64, sz = 2 : i64}, r = 0 : i64, stabilizer_basis = dense<[[false, false, false, true, true, false], [false, false, false, false, true, true]]> : tensor<2x6xi1>}
}
