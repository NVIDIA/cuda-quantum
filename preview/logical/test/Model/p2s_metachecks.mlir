// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2s"]} {
  fabric.code @redundant {
    distance = 3 : i64,
    hz = [array<i64: 0, 1>, array<i64: 1, 2>, array<i64: 0, 2>],
    k = 1 : i64,
    lx = [array<i64: 0, 1, 2>],
    lz = [array<i64: 0>],
    n = 3 : i64,
    partitions = {data = 3 : i64, sz = 3 : i64},
    r = 0 : i64
  }
  fabric.code_profile @redundant_profile {
    code = @redundant,
    metachecks = {z = dense<[[1, 1, 1]]> : tensor<1x3xi1>}
  }
}

// CHECK: fabric.code_profile @redundant_profile
// CHECK-SAME: code = @redundant
// CHECK-SAME: metachecks = {z = dense<
// CHECK-SAME: tensor<1x3xi1>}
