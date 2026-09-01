// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --prepare-quake-for-qlx %s | FileCheck %s --check-prefix=CLEAN --implicit-check-not=cc.undef --implicit-check-not=ub.poison
// RUN: qlx-opt --prepare-quake-for-qlx --convert-quake-to-qlx %s | FileCheck %s --check-prefix=P0 --implicit-check-not=quake. --implicit-check-not=cc.

module {
  func.func @__nvqpp__mlirgen__dead_loop_carry()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c3 = arith.constant 3 : i64
    %wire = quake.null_wire
    %dead = cc.undef i64
    %loop:3 = cc.loop while (
        (%iv = %c0, %unused = %dead, %iter = %wire)
        -> (i64, i64, !quake.wire)) {
      %condition = arith.cmpi ne, %iv, %c3 : i64
      cc.condition %condition(
          %iv, %unused, %iter : i64, i64, !quake.wire)
    } do {
    ^bb0(%iv: i64, %unused: i64, %iter: !quake.wire):
      %next_wire = quake.h %iter : (!quake.wire) -> !quake.wire
      cc.continue %iv, %unused, %next_wire : i64, i64, !quake.wire
    } step {
    ^bb0(%iv: i64, %unused: i64, %iter: !quake.wire):
      %next_iv = arith.addi %iv, %c1 : i64
      cc.continue %next_iv, %unused, %iter : i64, i64, !quake.wire
    }
    quake.sink %loop#2 : !quake.wire
    return
  }
}

// CLEAN-LABEL: func.func @__nvqpp__mlirgen__dead_loop_carry
// CLEAN: %[[Q:.*]] = quake.null_wire
// CLEAN: %[[LOOP:.*]]:2 = cc.loop while ((%[[IV:.*]] = %{{.*}}, %[[WIRE:.*]] = %[[Q]]) -> (i64, !quake.wire))
// CLEAN: cc.condition %{{.*}}(%[[IV]], %[[WIRE]] : i64, !quake.wire)
// CLEAN: ^bb0(%[[BODY_IV:.*]]: i64, %[[BODY_WIRE:.*]]: !quake.wire):
// CLEAN: %[[H:.*]] = quake.h %[[BODY_WIRE]]
// CLEAN: cc.continue %[[BODY_IV]], %[[H]] : i64, !quake.wire
// CLEAN: ^bb0(%[[STEP_IV:.*]]: i64, %[[STEP_WIRE:.*]]: !quake.wire):
// CLEAN: %[[NEXT:.*]] = arith.addi %[[STEP_IV]],
// CLEAN: cc.continue %[[NEXT]], %[[STEP_WIRE]] : i64, !quake.wire
// CLEAN: quake.sink %[[LOOP]]#1

// P0-LABEL: qlx.program @dead_loop_carry
// P0: %[[Q:.*]] = qlx.prepare "zero"
// P0: %[[REPEAT:.*]] = qlx.repeat 3
// P0-NEXT: iter(%[[ITER:.*]]: !qlx.logical_qubit = %[[Q]])
// P0: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[ITER]])
// P0: qlx.yield %[[H]]
// P0: qlx.discard %[[REPEAT]]
