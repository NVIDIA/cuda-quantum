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
  func.func @__nvqpp__mlirgen__folded_loop() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %q0 = quake.null_wire
    %loop:2 = cc.loop while (
        (%w = %q0, %iv = %c0) -> (!quake.wire, i32)) {
      // CUDA-Q's normalized counted-loop form may retain a forward signed
      // comparison rather than rewriting it to `ne`.
      %go = arith.cmpi slt, %iv, %c5 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      %t = quake.t %w : (!quake.wire) -> !quake.wire
      cc.continue %t, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %n = arith.addi %iv, %c1 : i32
      cc.continue %w, %n : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @folded_loop
// CHECK: %[[LOOP_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[LOOP_RESULT:.*]] = cflow.repeat 5
// CHECK: iter(%[[LOOP_ARG:.*]]: !qlx.logical_qubit = %[[LOOP_Q]])
// CHECK: %[[LOOP_T:.*]] = qlx.apply #qlx.action<t>(%[[LOOP_ARG]])
// CHECK: cflow.yield %[[LOOP_T]]
// CHECK: qlx.discard %[[LOOP_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__loop_local_measurement()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %carried = quake.null_wire
    %loop:2 = cc.loop while (
        (%w = %carried, %iv = %c0) -> (!quake.wire, i32)) {
      %go = arith.cmpi ne, %iv, %c3 : i32
      cc.condition %go(%w, %iv : !quake.wire, i32)
    } do {
    ^bb0(%w: !quake.wire, %iv: i32):
      %local = quake.null_wire
      %m, %measured = quake.mx %local
          : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
      %bit = quake.discriminate %m : (!cc.measure_handle) -> i1
      quake.sink %measured : !quake.wire
      %next = quake.h %w : (!quake.wire) -> !quake.wire
      cc.continue %next, %iv : !quake.wire, i32
    } step {
    ^bb0(%w: !quake.wire, %iv: i32):
      %next = arith.addi %iv, %c1 : i32
      cc.continue %w, %next : !quake.wire, i32
    }
    quake.sink %loop#0 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @loop_local_measurement
// CHECK: %[[LOCAL_CARRY:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[LOCAL_RESULT:.*]] = cflow.repeat 3
// CHECK: iter(%[[LOCAL_ARG:.*]]: !qlx.logical_qubit = %[[LOCAL_CARRY]])
// CHECK: %[[LOCAL_Q:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: qlx.measure <X> %[[LOCAL_Q]]
// CHECK: %[[LOCAL_H:.*]] = qlx.apply #qlx.action<h>(%[[LOCAL_ARG]])
// CHECK: cflow.yield %[[LOCAL_H]]
// CHECK: qlx.discard %[[LOCAL_RESULT]]

// -----

module {
  func.func @__nvqpp__mlirgen__constant_if() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %true = arith.constant true
    %0 = quake.null_wire
    %r = cc.if (%true) ((%a = %0)) -> (!quake.wire) {
      %h = quake.h %a : (!quake.wire) -> !quake.wire
      cc.continue %h : !quake.wire
    } else {
      cc.continue %a : !quake.wire
    }
    quake.sink %r : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @constant_if
// CHECK-NOT: cflow.if
// CHECK: %[[CONST_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[CONST_H:.*]] = qlx.apply #qlx.action<h>(%[[CONST_Q]])
// CHECK: qlx.discard %[[CONST_H]]
// CHECK-NOT: cflow.if

// -----

module {
  func.func @__nvqpp__mlirgen__adaptive() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %q0 = quake.null_wire
    %q1 = quake.null_wire
    %m, %w = quake.mz %q0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %cond = quake.discriminate %m : (!cc.measure_handle) -> i1
    %sel = cc.if (%cond) ((%c = %q1)) -> (!quake.wire) {
      %t = quake.t %c : (!quake.wire) -> !quake.wire
      cc.continue %t : !quake.wire
    } else {
      cc.continue %c : !quake.wire
    }
    quake.sink %w : !quake.wire
    quake.sink %sel : !quake.wire
    return %cond : i1
  }
}

// CHECK-LABEL: qlx.program @adaptive
// CHECK: %[[ADAPT_MEASURE_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[ADAPT_DATA_Q:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[ADAPT_M:.*]] = qlx.measure <Z> %[[ADAPT_MEASURE_Q]]
// CHECK: %[[ADAPT_IF:.*]] = cflow.if %[[ADAPT_M]]
// CHECK: %[[ADAPT_T:.*]] = qlx.apply #qlx.action<t>(%[[ADAPT_DATA_Q]])
// CHECK: cflow.yield %[[ADAPT_T]]
// CHECK: cflow.yield %[[ADAPT_DATA_Q]]
// CHECK: qlx.discard %[[ADAPT_IF]]
// CHECK: qlx.return %[[ADAPT_M]]
