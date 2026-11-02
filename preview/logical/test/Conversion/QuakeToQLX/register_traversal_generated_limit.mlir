// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: not qlx-opt '--expand-quake-register-traversals=maximum-generated-operations=2' %s 2>&1 | FileCheck %s

module {
  func.func @over_generated_operation_limit() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c3 = arith.constant 3 : i64
    %q = quake.alloca !quake.veq<3>
    %loop = cc.loop while ((%iv = %c0) -> (i64)) {
      %go = arith.cmpi slt, %iv, %c3 : i64
      cc.condition %go(%iv : i64)
    } do {
    ^bb0(%iv: i64):
      %ref = quake.extract_ref %q[%iv] : (!quake.veq<3>, i64) -> !quake.ref
      quake.h %ref : (!quake.ref) -> ()
      cc.continue %iv : i64
    } step {
    ^bb0(%iv: i64):
      %next = arith.addi %iv, %c1 : i64
      cc.continue %next : i64
    }
    return
  }
}

// CHECK: error: 'cc.loop' op wire-blocking register traversal would exceed the 2 generated-operation preparation limit
