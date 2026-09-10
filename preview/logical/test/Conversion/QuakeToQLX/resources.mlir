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
  func.func @__nvqpp__mlirgen__t_and_tdg() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %t = quake.t %0 : (!quake.wire) -> !quake.wire
    %tdg = quake.t<adj> %1 : (!quake.wire) -> !quake.wire
    quake.sink %t : !quake.wire
    quake.sink %tdg : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @t_and_tdg
// CHECK: %[[T_Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[T_Q1:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[T_RESULT:.*]] = qlx.apply #qlx.action<t>(%[[T_Q0]])
// CHECK: %[[TDG_RESULT:.*]] = qlx.apply #qlx.action<tdg>(%[[T_Q1]])
// CHECK: qlx.discard %[[T_RESULT]], %[[TDG_RESULT]]
// -----

module {
  func.func @__nvqpp__mlirgen__ccz() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    %b = quake.null_wire
    %t = quake.null_wire
    %r:3 = quake.z [%a, %b] %t
        : (!quake.wire, !quake.wire, !quake.wire)
       -> (!quake.wire, !quake.wire, !quake.wire)
    quake.sink %r#0 : !quake.wire
    quake.sink %r#1 : !quake.wire
    quake.sink %r#2 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @ccz
// CHECK-NOT: qlx.apply #qlx.action<h>
// CHECK: %[[CCZ_Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[CCZ_Q1:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[CCZ_Q2:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[CCZ_RESULT:.*]]:3 = qlx.apply #qlx.action<ccz>(%[[CCZ_Q0]], %[[CCZ_Q1]], %[[CCZ_Q2]])
// CHECK: qlx.discard %[[CCZ_RESULT]]#0, %[[CCZ_RESULT]]#1, %[[CCZ_RESULT]]#2
// CHECK-NOT: qlx.apply #qlx.action<h>

// -----

module {
  func.func @__nvqpp__mlirgen__toffoli() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    %b = quake.null_wire
    %t = quake.null_wire
    %r:3 = quake.x [%a, %b] %t
        : (!quake.wire, !quake.wire, !quake.wire)
       -> (!quake.wire, !quake.wire, !quake.wire)
    quake.sink %r#0 : !quake.wire
    quake.sink %r#1 : !quake.wire
    quake.sink %r#2 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @toffoli
// CHECK: %[[TOF_Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[TOF_Q1:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[TOF_Q2:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[TOF_CCX:.*]]:3 = qlx.apply #qlx.action<ccx>(%[[TOF_Q0]], %[[TOF_Q1]], %[[TOF_Q2]])
// CHECK: qlx.discard %[[TOF_CCX]]#0, %[[TOF_CCX]]#1, %[[TOF_CCX]]#2
