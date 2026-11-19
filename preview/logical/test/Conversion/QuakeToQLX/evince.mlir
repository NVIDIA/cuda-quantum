// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake.

// Upstream aggregate expansion and memtoreg produce one variadic evince.
// Each result must retain the corresponding live owner, including its order.
module {
  func.func @__nvqpp__mlirgen__live_lifetimes()
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    %b = quake.null_wire
    %logged:2 = quake.evince %b, %a
        : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire) {compilerGenerated}
    %h = quake.h %logged#0 : (!quake.wire) -> !quake.wire
    quake.sink %h : !quake.wire
    quake.sink %logged#1 : !quake.wire
    return
  }
}

// CHECK-LABEL: qlx.program @live_lifetimes
// CHECK: %[[A:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[B:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[H:.*]] = qlx.apply #qlx.action<h>(%[[B]])
// CHECK: qlx.discard %[[H]], %[[A]]
// CHECK-NEXT: }

// -----

// Measured wires forward only their lifetime state. Sinking them must not
// discard the logical owner already consumed by destructive measurement.
module {
  func.func @__nvqpp__mlirgen__mixed_lifetimes() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %a = quake.null_wire
    %b = quake.null_wire
    %c = quake.null_wire
    %ma, %wa = quake.mz %a : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %mc, %wc = quake.mz %c : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %logged:3 = quake.evince %wa, %b, %wc
        : (!quake.wire, !quake.wire, !quake.wire)
       -> (!quake.wire, !quake.wire, !quake.wire) {compilerGenerated}
    %again = quake.evince %logged#0
        : (!quake.wire) -> !quake.wire {compilerGenerated}
    %bit = quake.discriminate %ma : (!cc.measure_handle) -> i1
    quake.sink %again : !quake.wire
    quake.sink %logged#1 : !quake.wire
    quake.sink %logged#2 : !quake.wire
    return %bit : i1
  }
}

// CHECK-LABEL: qlx.program @mixed_lifetimes
// CHECK: %[[A:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[B:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[C:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[BIT:.*]] = qlx.measure <Z> %[[A]]
// CHECK: qlx.measure <Z> %[[C]]
// CHECK-NEXT: qlx.discard %[[B]] : !qlx.logical_qubit
// CHECK-NEXT: qlx.return %[[BIT]] : i1
