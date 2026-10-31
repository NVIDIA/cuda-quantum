// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2s"]} {
  // A stabilizer-state code has no protected logical port.  It is a native
  // code artifact, not a k=1 code padded with a fictional spectator logical.
  fabric.code @prepared_zero {
    distance = 0 : i64,
    hz = [array<i64: 0>],
    k = 0 : i64,
    lx = [],
    lz = [],
    metadata = {distance_status = "unknown"},
    n = 1 : i64,
    partitions = {data = 1 : i64},
    r = 0 : i64
  }
  fabric.code_profile @prepared_zero_profile {
    code = @prepared_zero,
    distance_status = "unknown"
  }
  fabric.encoding @prepared_zero_encoding {
    block = "state",
    code = @prepared_zero,
    logical_ports = [],
    profile = @prepared_zero_profile
  }
}

// CHECK: fabric.code @prepared_zero
// CHECK-SAME: k = 0 : i64
// CHECK: fabric.encoding @prepared_zero_encoding
// CHECK-SAME: logical_ports = []
