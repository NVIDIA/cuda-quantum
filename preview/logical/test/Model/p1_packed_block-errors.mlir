// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s 2>&1 | FileCheck %s

// The legacy P1 encoded-block operation is deliberately no longer registered.
module attributes {qlx.profiles = ["p1"]} {
  lvm.domain @vm {
    lvm.space @memory {capabilities = [], capacity = 1 : i64}
    "lvm.block"() {
      sym_name = "b0",
      space = @memory,
      slot = 0 : i64,
      encoding = @some_encoding,
      logical_capacity = 2 : i64
    } : () -> ()
  }
}

// CHECK: error: unregistered operation 'lvm.block' found in dialect ('lvm') that does not allow unknown operations
