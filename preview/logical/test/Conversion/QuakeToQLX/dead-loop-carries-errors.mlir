// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --prune-dead-cc-loop-carries --split-input-file --verify-diagnostics %s

module {
  func.func @loop_with_break() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c3 = arith.constant 3 : i64
    %wire = quake.null_wire
    // expected-error@+1 {{'cc.loop' op cannot prune dead carries: break is outside the supported subset}}
    %loop:2 = cc.loop while (
        (%iv = %c0, %iter = %wire) -> (i64, !quake.wire)) {
      %condition = arith.cmpi ne, %iv, %c3 : i64
      cc.condition %condition(%iv, %iter : i64, !quake.wire)
    } do {
    ^bb0(%iv: i64, %iter: !quake.wire):
      cc.break %iv, %iter : i64, !quake.wire
    } step {
    ^bb0(%iv: i64, %iter: !quake.wire):
      %next_iv = arith.addi %iv, %c1 : i64
      cc.continue %next_iv, %iter : i64, !quake.wire
    }
    quake.sink %loop#1 : !quake.wire
    return
  }
}

// -----

module {
  func.func @loop_inside_scope() {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %wire = quake.null_wire
    %scoped = cc.scope -> !quake.wire {
      %dead = cc.undef i64
      // expected-error@+1 {{'cc.loop' op cannot prune dead carries while cc.scope remains; lower the scope before the Quake-to-P0 handoff}}
      %loop:3 = cc.loop while (
          (%iv = %c0, %unused = %dead, %iter = %wire)
          -> (i64, i64, !quake.wire)) {
        %condition = arith.cmpi ne, %iv, %c1 : i64
        cc.condition %condition(
            %iv, %unused, %iter : i64, i64, !quake.wire)
      } do {
      ^bb0(%iv: i64, %unused: i64, %iter: !quake.wire):
        cc.continue %iv, %unused, %iter : i64, i64, !quake.wire
      } step {
      ^bb0(%iv: i64, %unused: i64, %iter: !quake.wire):
        %next_iv = arith.addi %iv, %c1 : i64
        cc.continue %next_iv, %unused, %iter : i64, i64, !quake.wire
      }
      cc.continue %loop#2 : !quake.wire
    }
    quake.sink %scoped : !quake.wire
    return
  }
}
