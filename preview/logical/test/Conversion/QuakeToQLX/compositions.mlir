// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake. --implicit-check-not=qlx.resource_request --implicit-check-not=qlx.consume_resource

module {
  func.func @__nvqpp__mlirgen__arithmetic_slice() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %control = quake.null_wire
    %a0 = quake.null_wire
    %a1 = quake.null_wire
    %a2 = quake.null_wire
    %flag = quake.null_wire
    %control_h = quake.h %control : (!quake.wire) -> !quake.wire
    %flag_h = quake.h %flag : (!quake.wire) -> !quake.wire
    %rounds:5 = cc.loop while (
        (%ctrl = %control_h, %x0 = %a0, %x1 = %a1, %x2 = %a2, %iv = %c0)
        -> (!quake.wire, !quake.wire, !quake.wire, !quake.wire, i32)) {
      %keep_going = arith.cmpi ne, %iv, %c4 : i32
      cc.condition %keep_going(
          %ctrl, %x0, %x1, %x2, %iv
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire, i32)
    } do {
    ^bb0(%ctrl: !quake.wire, %x0: !quake.wire, %x1: !quake.wire,
         %x2: !quake.wire, %iv: i32):
      %fanout:2 = quake.x [%ctrl] %x0
          : (!quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire)
      %carry0:3 = quake.x [%fanout#0, %fanout#1] %x1
          : (!quake.wire, !quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire, !quake.wire)
      %carry1:3 = quake.x [%carry0#0, %carry0#2] %x2
          : (!quake.wire, !quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire, !quake.wire)
      %theta = arith.constant 1.250000e-01 : f64
      %phase = quake.rz (%theta) %carry1#2
          : (f64, !quake.wire) -> !quake.wire
      %phase_t = quake.t %phase : (!quake.wire) -> !quake.wire
      %swapped:2 = quake.swap %carry0#1, %carry1#1
          : (!quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire)
      %next_ctrl = quake.h %carry1#0 : (!quake.wire) -> !quake.wire
      cc.continue %next_ctrl, %swapped#0, %swapped#1, %phase_t, %iv
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire, i32
    } step {
    ^bb0(%ctrl: !quake.wire, %x0: !quake.wire, %x1: !quake.wire,
         %x2: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %ctrl, %x0, %x1, %x2, %next
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire, i32
    }
    %measurement, %measured_flag = quake.mz %flag_h
        : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %condition = quake.discriminate %measurement
        : (!cc.measure_handle) -> i1
    %corrected:4 = cc.if (%condition)
        ((%q0 = %rounds#0, %q1 = %rounds#1,
          %q2 = %rounds#2, %q3 = %rounds#3))
        -> (!quake.wire, !quake.wire, !quake.wire, !quake.wire) {
      %x = quake.x %q0 : (!quake.wire) -> !quake.wire
      %cz:2 = quake.z [%q1] %q2
          : (!quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire)
      cc.continue %x, %cz#0, %cz#1, %q3
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire
    } else {
      %cx:2 = quake.x [%q0] %q1
          : (!quake.wire, !quake.wire)
         -> (!quake.wire, !quake.wire)
      %y = quake.y %q3 : (!quake.wire) -> !quake.wire
      cc.continue %cx#0, %cx#1, %q2, %y
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire
    }
    quake.sink %measured_flag : !quake.wire
    quake.sink %corrected#0 : !quake.wire
    quake.sink %corrected#1 : !quake.wire
    quake.sink %corrected#2 : !quake.wire
    quake.sink %corrected#3 : !quake.wire
    return %condition : i1
  }
}

// CHECK-LABEL: qlx.program @arithmetic_slice
// CHECK: %[[AR_Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[AR_Q1:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[AR_Q2:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[AR_Q3:.*]] = qlx.prepare "zero" {allocation = 3 : i64
// CHECK: %[[AR_FLAG:.*]] = qlx.prepare "zero" {allocation = 4 : i64
// CHECK: %[[AR_CTRL_H:.*]] = qlx.apply #qlx.action<h>(%[[AR_Q0]])
// CHECK: %[[AR_FLAG_H:.*]] = qlx.apply #qlx.action<h>(%[[AR_FLAG]])
// CHECK: %[[AR_REPEAT:.*]]:4 = cflow.repeat 4
// CHECK: iter(%[[AR_CTRL:.*]]: !qlx.logical_qubit = %[[AR_CTRL_H]],
// CHECK: %[[AR_A0:.*]]: !qlx.logical_qubit = %[[AR_Q1]],
// CHECK: %[[AR_A1:.*]]: !qlx.logical_qubit = %[[AR_Q2]],
// CHECK: %[[AR_A2:.*]]: !qlx.logical_qubit = %[[AR_Q3]])
// CHECK: %[[AR_FANOUT:.*]]:2 = qlx.apply #qlx.action<cx>(%[[AR_CTRL]], %[[AR_A0]])
// CHECK: %[[AR_CCX0:.*]]:3 = qlx.apply #qlx.action<ccx>(%[[AR_FANOUT]]#0, %[[AR_FANOUT]]#1, %[[AR_A1]])
// CHECK: %[[AR_CCX1:.*]]:3 = qlx.apply #qlx.action<ccx>(%[[AR_CCX0]]#0, %[[AR_CCX0]]#2, %[[AR_A2]])
// CHECK: %[[AR_ANGLE:.*]] = arith.constant 1.250000e-01 : f64
// CHECK: %[[AR_ROT:.*]] = qlx.apply #qlx.action<pauli_rotation>(%[[AR_CCX1]]#2, %[[AR_ANGLE]]) {parameters = {sign = 1 : i64, x_mask = 0 : i64, z_mask = 1 : i64}}
// CHECK: %[[AR_T:.*]] = qlx.apply #qlx.action<t>(%[[AR_ROT]])
// CHECK: %[[AR_SWAP0:.*]]:2 = qlx.apply #qlx.action<cx>(%[[AR_CCX0]]#1, %[[AR_CCX1]]#1)
// CHECK: %[[AR_SWAP1:.*]]:2 = qlx.apply #qlx.action<cx>(%[[AR_SWAP0]]#1, %[[AR_SWAP0]]#0)
// CHECK: %[[AR_SWAP2:.*]]:2 = qlx.apply #qlx.action<cx>(%[[AR_SWAP1]]#1, %[[AR_SWAP1]]#0)
// CHECK: %[[AR_NEXT_CTRL:.*]] = qlx.apply #qlx.action<h>(%[[AR_CCX1]]#0)
// CHECK: cflow.yield %[[AR_NEXT_CTRL]], %[[AR_SWAP2]]#0, %[[AR_SWAP2]]#1, %[[AR_T]]
// CHECK: %[[AR_MEAS:.*]] = qlx.measure <Z> %[[AR_FLAG_H]]
// CHECK: %[[AR_IF:.*]]:4 = cflow.if %[[AR_MEAS]]
// CHECK: %[[AR_THEN_X:.*]] = qlx.apply #qlx.action<x>(%[[AR_REPEAT]]#0)
// CHECK: %[[AR_THEN_CZ:.*]]:2 = qlx.apply #qlx.action<cz>(%[[AR_REPEAT]]#1, %[[AR_REPEAT]]#2)
// CHECK: cflow.yield %[[AR_THEN_X]], %[[AR_THEN_CZ]]#0, %[[AR_THEN_CZ]]#1, %[[AR_REPEAT]]#3
// CHECK: %[[AR_ELSE_CX:.*]]:2 = qlx.apply #qlx.action<cx>(%[[AR_REPEAT]]#0, %[[AR_REPEAT]]#1)
// CHECK: %[[AR_ELSE_Y:.*]] = qlx.apply #qlx.action<y>(%[[AR_REPEAT]]#3)
// CHECK: cflow.yield %[[AR_ELSE_CX]]#0, %[[AR_ELSE_CX]]#1, %[[AR_REPEAT]]#2, %[[AR_ELSE_Y]]
// CHECK: qlx.discard %[[AR_IF]]#0, %[[AR_IF]]#1, %[[AR_IF]]#2, %[[AR_IF]]#3
// CHECK: qlx.return %[[AR_MEAS]]

// -----

module {
  func.func @__nvqpp__mlirgen__multi_wire_adaptive() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %condition_qubit = quake.null_wire
    %q0 = quake.null_wire
    %q1 = quake.null_wire
    %q2 = quake.null_wire
    %q3 = quake.null_wire
    %measurement, %measured_wire = quake.mz %condition_qubit
        : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %condition = quake.discriminate %measurement
        : (!cc.measure_handle) -> i1
    %selected:4 = cc.if (%condition)
        ((%a = %q0, %b = %q1, %c = %q2, %d = %q3))
        -> (!quake.wire, !quake.wire, !quake.wire, !quake.wire) {
      %ab:2 = quake.x [%a] %b
          : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %cd:2 = quake.x [%c] %d
          : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      cc.continue %ab#0, %ab#1, %cd#0, %cd#1
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire
    } else {
      %ac:2 = quake.x [%a] %c
          : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      %bd:2 = quake.x [%b] %d
          : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
      cc.continue %ac#0, %bd#0, %ac#1, %bd#1
          : !quake.wire, !quake.wire, !quake.wire, !quake.wire
    }
    quake.sink %measured_wire : !quake.wire
    quake.sink %selected#0 : !quake.wire
    quake.sink %selected#1 : !quake.wire
    quake.sink %selected#2 : !quake.wire
    quake.sink %selected#3 : !quake.wire
    return %condition : i1
  }
}

// CHECK-LABEL: qlx.program @multi_wire_adaptive
// CHECK: %[[MW_COND_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[MW_Q0:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[MW_Q1:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[MW_Q2:.*]] = qlx.prepare "zero" {allocation = 3 : i64
// CHECK: %[[MW_Q3:.*]] = qlx.prepare "zero" {allocation = 4 : i64
// CHECK: %[[MW_MEAS:.*]] = qlx.measure <Z> %[[MW_COND_Q]]
// CHECK: %[[MW_IF:.*]]:4 = cflow.if %[[MW_MEAS]]
// CHECK: %[[MW_THEN_01:.*]]:2 = qlx.apply #qlx.action<cx>(%[[MW_Q0]], %[[MW_Q1]])
// CHECK: %[[MW_THEN_23:.*]]:2 = qlx.apply #qlx.action<cx>(%[[MW_Q2]], %[[MW_Q3]])
// CHECK: cflow.yield %[[MW_THEN_01]]#0, %[[MW_THEN_01]]#1, %[[MW_THEN_23]]#0, %[[MW_THEN_23]]#1
// CHECK: %[[MW_ELSE_02:.*]]:2 = qlx.apply #qlx.action<cx>(%[[MW_Q0]], %[[MW_Q2]])
// CHECK: %[[MW_ELSE_13:.*]]:2 = qlx.apply #qlx.action<cx>(%[[MW_Q1]], %[[MW_Q3]])
// CHECK: cflow.yield %[[MW_ELSE_02]]#0, %[[MW_ELSE_13]]#0, %[[MW_ELSE_02]]#1, %[[MW_ELSE_13]]#1
// CHECK: qlx.discard %[[MW_IF]]#0, %[[MW_IF]]#1, %[[MW_IF]]#2, %[[MW_IF]]#3
// CHECK: qlx.return %[[MW_MEAS]]
