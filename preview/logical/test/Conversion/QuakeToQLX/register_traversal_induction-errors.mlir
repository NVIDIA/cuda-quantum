// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: not qlx-opt --expand-quake-register-traversals %s 2>&1 | FileCheck %s

module {
  // Resetting the induction in the body makes this source loop nonterminating.
  func.func @body_resets_induction() {
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
      cc.continue %c0 : i64
    } step {
    ^bb0(%iv: i64):
      %next = arith.addi %iv, %c1 : i64
      cc.continue %next : i64
    }
    return
  }

  // Advancing in both body and step terminates, but has a different trip count.
  func.func @body_advances_induction() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c4 = arith.constant 4 : i64
    %q = quake.alloca !quake.veq<8>
    %loop = cc.loop while ((%iv = %c0) -> (i64)) {
      %go = arith.cmpi slt, %iv, %c4 : i64
      cc.condition %go(%iv : i64)
    } do {
    ^bb0(%iv: i64):
      %ref = quake.extract_ref %q[%iv] : (!quake.veq<8>, i64) -> !quake.ref
      quake.x %ref : (!quake.ref) -> ()
      %body_next = arith.addi %iv, %c1 : i64
      cc.continue %body_next : i64
    } step {
    ^bb0(%iv: i64):
      %next = arith.addi %iv, %c1 : i64
      cc.continue %next : i64
    }
    return
  }

  // The mathematical induction crosses the bound, but i64 addi wraps and the
  // signed source loop continues instead of terminating after one iteration.
  func.func @induction_step_wraps() {
    %start = arith.constant 9223372036854775806 : i64
    %stop = arith.constant 9223372036854775807 : i64
    %c2 = arith.constant 2 : i64
    %q = quake.alloca !quake.veq<3>
    %loop = cc.loop while ((%iv = %start) -> (i64)) {
      %go = arith.cmpi slt, %iv, %stop : i64
      cc.condition %go(%iv : i64)
    } do {
    ^bb0(%iv: i64):
      %index = arith.andi %iv, %c2 : i64
      %ref = quake.extract_ref %q[%index]
          : (!quake.veq<3>, i64) -> !quake.ref
      quake.y %ref : (!quake.ref) -> ()
      cc.continue %iv : i64
    } step {
    ^bb0(%iv: i64):
      %next = arith.addi %iv, %c2 : i64
      cc.continue %next : i64
    }
    return
  }
}

// CHECK-COUNT-3: error: 'cc.loop' op wire-blocking register traversal is not a supported normalized constant counted loop
