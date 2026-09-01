// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

// Regression: canonical P1 capacity and placement are logical-owner concepts.
// A later P2 lowering may encode these two owners into one high-rate block,
// but P1 itself contains no code, encoding, block, or logical-port index.
module attributes {qlx.profiles = ["p1"]} {
  lvm.domain @vm {
    lvm.space @memory {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">],
      capacity = 2 : i64
    }
  }
}

// CHECK: lvm.domain @vm
// CHECK: lvm.space @memory
// CHECK-SAME: capacity = 2 : i64
// CHECK-NOT: fabric.code
// CHECK-NOT: fabric.encoding
// CHECK-NOT: lvm.block
