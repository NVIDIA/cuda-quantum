// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: not qlx-opt %s -split-input-file 2>&1 | FileCheck %s

module {
  qlx.program @bad_profile : () -> () attributes {qlx.profile = "p1"} {
    qlx.return
  }
}

// CHECK: error: 'qlx.program' op requires qlx.stage = "p0"

// -----

module {
  qlx.program @unknown_action : (!qlx.logical_qubit) -> (!qlx.logical_qubit)
      attributes {qlx.profile = "p0"} {
  ^bb0(%q: !qlx.logical_qubit):
    %out = qlx.apply @missing(%q) : (!qlx.logical_qubit) -> (!qlx.logical_qubit)
    qlx.return %out : !qlx.logical_qubit
  }
}

// CHECK: error: 'qlx.apply' op references unknown qlx.action @missing

// -----

module {
  qlx.action @bad_clifford : (!qlx.logical_qubit) -> !qlx.logical_qubit {
    kind = "composite",
    clifford_action = #qlx.clifford_action<
      matrix = [1, 0, 1, 0], phases = [0, 0], ports = ["q"]>
  }
}

// CHECK: error: matrix does not preserve the binary symplectic form

// -----

module {
  qlx.objective_body @runtime_objective : () -> () attributes {
      objective_kind = "action", qlx.stage = "p0"} {
    %event = qlx.resource_request "t_state"
      : !event.handle<!qlx.logical_resource<"t_state">, "linear">
    qlx.return
  }
}

// CHECK: error: runtime and resource orchestration is not legal inside a closed logical objective

// -----

module {
  qlx.program @fixed_action_parameters : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %h = qlx.apply #qlx.action<h>(%q) {parameters = {unexpected = 1 : i64}}
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %h : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK: error: 'qlx.apply' op fixed built-in actions do not accept parameter bindings

// -----

module {
  qlx.program @fixed_action_classical_input : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %flag = arith.constant true
    %h = qlx.apply #qlx.action<h>(%q, %flag)
        : (!qlx.logical_qubit, i1) -> !qlx.logical_qubit
    %m = qlx.measure #qlx.pauli<Z> %h : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK: error: 'qlx.apply' op fixed built-in action requires exactly 1 logical-qubit inputs and results and no classical payloads

// -----

module {
  qlx.program @fixed_action_classical_result : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %t:2 = qlx.apply #qlx.action<t>(%q)
        : (!qlx.logical_qubit) -> (!qlx.logical_qubit, i1)
    qlx.discard %t#0 : !qlx.logical_qubit
    qlx.return %t#1 : i1
  }
}

// CHECK: error: 'qlx.apply' op fixed built-in action requires exactly 1 logical-qubit inputs and results and no classical payloads

// -----

module {
  qlx.objective_body @embedded_event_objective :
      (!event.handle<!qlx.logical_resource<"t_state">, "linear">) -> ()
      attributes {objective_kind = "action", qlx.stage = "p0"} {
  ^bb0(%event: !event.handle<!qlx.logical_resource<"t_state">, "linear">):
    %ready = event.test %event : !event.handle<!qlx.logical_resource<"t_state">, "linear"> -> i1
    qlx.return
  }
}

// CHECK: error: runtime and resource orchestration is not legal inside a closed logical objective
