// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.stages = ["p0"], qlx.facets = []} {
  qlx.objective_body @double_h : (!qlx.logical_qubit) -> !qlx.logical_qubit
      attributes {objective_kind = "action", qlx.stage = "p0"} {
  ^bb0(%q: !qlx.logical_qubit):
    %q1 = qlx.apply #qlx.action<h>(%q)
      : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %q2 = qlx.apply #qlx.action<h>(%q1)
      : (!qlx.logical_qubit) -> !qlx.logical_qubit
    qlx.return %q2 : !qlx.logical_qubit
  }
  qlx.action @double_h_action : (!qlx.logical_qubit) -> !qlx.logical_qubit {
    kind = "composite", semantics = @double_h
  }
}

// CHECK: qlx.objective_body @double_h
// CHECK-SAME: objective_kind = "action"
// CHECK-NOT: qlx.program @double_h
// CHECK: qlx.action @double_h_action
// CHECK-SAME: semantics = @double_h
