// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {
  qlx.model_version = "0.3.10-proposed",
  qlx.ir_version = "0.4-draft",
  qlx.profiles = ["p0"]
} {
  qlx.program @one : () -> (i1) attributes {qlx.profile = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %q1 = qlx.apply #qlx.action<h>(%q0) : (!qlx.logical_qubit) -> (!qlx.logical_qubit)
    %m = qlx.measure <Z> %q1 : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK-NOT: qlx.instrument_decl
// CHECK-NOT: qlx.action @
// CHECK: qlx.program @one : () -> i1 attributes {qlx.profile = "p0"}
// CHECK: %[[Q0:.*]] = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
// CHECK: %[[Q1:.*]] = qlx.apply #qlx.action<h>(%[[Q0]]) : (!qlx.logical_qubit) -> !qlx.logical_qubit
// CHECK: %[[M:.*]] = qlx.measure <Z> %[[Q1]] : !qlx.logical_qubit -> i1
// CHECK: qlx.return %[[M]] : i1
