// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake.

module {
  func.func @__nvqpp__mlirgen__basic_gates() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %3 = quake.null_wire
    %4 = quake.null_wire
    %5 = quake.null_wire
    %6 = quake.null_wire
    %7 = quake.null_wire
    %8 = quake.null_wire
    %h = quake.h %0 : (!quake.wire) -> !quake.wire
    %s = quake.s %1 : (!quake.wire) -> !quake.wire
    %sdg = quake.s<adj> %2 : (!quake.wire) -> !quake.wire
    %x = quake.x %3 : (!quake.wire) -> !quake.wire
    %y = quake.y %4 : (!quake.wire) -> !quake.wire
    %z = quake.z %5 : (!quake.wire) -> !quake.wire
    %cx:2 = quake.x [%6] %7 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    %cz:2 = quake.z [%cx#0] %8 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.sink %h : !quake.wire
    quake.sink %s : !quake.wire
    quake.sink %sdg : !quake.wire
    quake.sink %x : !quake.wire
    quake.sink %y : !quake.wire
    quake.sink %z : !quake.wire
    quake.sink %cx#1 : !quake.wire
    quake.sink %cz#0 : !quake.wire
    quake.sink %cz#1 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @basic_gates
// CHECK: %[[BG_Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[BG_Q1:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[BG_Q2:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[BG_Q3:.*]] = qlx.prepare "zero" {allocation = 3 : i64
// CHECK: %[[BG_Q4:.*]] = qlx.prepare "zero" {allocation = 4 : i64
// CHECK: %[[BG_Q5:.*]] = qlx.prepare "zero" {allocation = 5 : i64
// CHECK: %[[BG_Q6:.*]] = qlx.prepare "zero" {allocation = 6 : i64
// CHECK: %[[BG_Q7:.*]] = qlx.prepare "zero" {allocation = 7 : i64
// CHECK: %[[BG_Q8:.*]] = qlx.prepare "zero" {allocation = 8 : i64
// CHECK: %[[BG_H:.*]] = qlx.apply #qlx.action<h>(%[[BG_Q0]])
// CHECK: %[[BG_S:.*]] = qlx.apply #qlx.action<s>(%[[BG_Q1]])
// CHECK: %[[BG_SDG:.*]] = qlx.apply #qlx.action<sdg>(%[[BG_Q2]])
// CHECK: %[[BG_X:.*]] = qlx.apply #qlx.action<x>(%[[BG_Q3]])
// CHECK: %[[BG_Y:.*]] = qlx.apply #qlx.action<y>(%[[BG_Q4]])
// CHECK: %[[BG_Z:.*]] = qlx.apply #qlx.action<z>(%[[BG_Q5]])
// CHECK: %[[BG_CX:.*]]:2 = qlx.apply #qlx.action<cx>(%[[BG_Q6]], %[[BG_Q7]])
// CHECK: %[[BG_CZ:.*]]:2 = qlx.apply #qlx.action<cz>(%[[BG_CX]]#0, %[[BG_Q8]])
// CHECK: qlx.discard %[[BG_H]], %[[BG_S]], %[[BG_SDG]], %[[BG_X]], %[[BG_Y]], %[[BG_Z]], %[[BG_CX]]#1, %[[BG_CZ]]#0, %[[BG_CZ]]#1

// -----

module {
  func.func @__nvqpp__mlirgen__swap() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %r:2 = quake.swap %0, %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
    quake.sink %r#0 : !quake.wire
    quake.sink %r#1 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @swap
// CHECK: %[[SW_Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[SW_Q1:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[SW_CX0:.*]]:2 = qlx.apply #qlx.action<cx>(%[[SW_Q0]], %[[SW_Q1]])
// CHECK: %[[SW_CX1:.*]]:2 = qlx.apply #qlx.action<cx>(%[[SW_CX0]]#1, %[[SW_CX0]]#0)
// CHECK: %[[SW_CX2:.*]]:2 = qlx.apply #qlx.action<cx>(%[[SW_CX1]]#1, %[[SW_CX1]]#0)
// CHECK: qlx.discard %[[SW_CX2]]#0, %[[SW_CX2]]#1
// CHECK-NOT: qlx.apply #qlx.action<cx>
