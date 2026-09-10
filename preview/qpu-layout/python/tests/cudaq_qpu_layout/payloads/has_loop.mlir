// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// A payload with a surviving `cc.loop`. The layout simulator costs a schedule,
// which it cannot do for a loop it has not seen unrolled, so this must be
// rejected with a clear diagnostic rather than traced approximately.

quake.wire_set @wires[2]

func.func @has_loop() attributes {"cudaq-entrypoint"} {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %q = quake.borrow_wire @wires[0] : !quake.wire

  %r:2 = cc.loop while ((%i = %c0, %w = %q) -> (i64, !quake.wire)) {
    %cond = arith.cmpi slt, %i, %c4 : i64
    cc.condition %cond (%i, %w : i64, !quake.wire)
  } do {
  ^bb0(%i: i64, %w: !quake.wire):
    %wh = quake.h %w : (!quake.wire) -> !quake.wire
    cc.continue %i, %wh : i64, !quake.wire
  } step {
  ^bb0(%i: i64, %w: !quake.wire):
    %next = arith.addi %i, %c1 : i64
    cc.continue %next, %w : i64, !quake.wire
  }

  quake.return_wire %r#1 : !quake.wire
  return
}
