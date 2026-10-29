// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s
// RUN: not qlx-opt %s --fabric-count='root=qec' 2>&1 | FileCheck %s --check-prefix=COUNT-ERR

// `while`'s round-trip is the shared `cflow` dialect's own concern (see
// test/Dialect/Cflow/roundtrip-while.mlir); this file exists for
// `fabric-count`'s rejection of it (Non-Goal: no analytic support for
// dynamic loops), which is a real pass-behavior check the Cflow suite
// does not cover.

module attributes {qlx.profiles = ["p2n"]} {
  fabric.protocol @qec : (i1) -> i1 {
  ^bb0(%go: i1):
    %0 = "cflow.while"(%go) <{max_iterations = 8 : i64}> ({
    ^bb0(%current: i1):
      "cflow.while_condition"(%current, %current) : (i1, i1) -> ()
    }, {
    ^bb0(%current: i1):
      cflow.yield %current : i1
    }) : (i1) -> i1
    fabric.protocol_return %0 : i1
  }
}

// CHECK: cflow.while
// CHECK: cflow.while_condition
// COUNT-ERR: fabric-count does not support this dynamic or unrecognized region-bearing executable operation
