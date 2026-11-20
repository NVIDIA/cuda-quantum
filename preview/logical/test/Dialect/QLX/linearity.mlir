// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --verify-linear-ownership | FileCheck %s

// Legal single-ownership patterns. The P0 verifiers and the
// verify-linear-ownership pass both accept them.

module {
  // Each value is consumed once; measurement ends the chain.
  // CHECK-LABEL: qlx.program @straight_line
  qlx.program @straight_line : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" : !qlx.logical_qubit
    %h = qlx.apply #qlx.action<h>(%q0) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %c, %t = qlx.apply #qlx.action<cx>(%h, %q1)
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    %m = qlx.measure <Z> %c : !qlx.logical_qubit -> i1
    qlx.discard %t : !qlx.logical_qubit
    qlx.return %m : i1
  }

  // The Pauli-byproduct pattern consumes the same captured value once in
  // each exclusive branch.
  // CHECK-LABEL: qlx.program @both_branches
  qlx.program @both_branches : (i1) -> i1 attributes {qlx.stage = "p0"} {
  ^bb0(%c: i1):
    %q = qlx.prepare "zero" : !qlx.logical_qubit
    %r = cflow.if %c -> !qlx.logical_qubit {
      %x = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      cflow.yield %x : !qlx.logical_qubit
    } else {
      cflow.yield %q : !qlx.logical_qubit
    }
    %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }

  // A value captured into nested ifs is consumed on all four paths.
  // CHECK-LABEL: qlx.program @nested_branches
  qlx.program @nested_branches : (i1, i1) -> i1 attributes {qlx.stage = "p0"} {
  ^bb0(%a: i1, %b: i1):
    %q = qlx.prepare "zero" : !qlx.logical_qubit
    %r = cflow.if %a -> !qlx.logical_qubit {
      %inner = cflow.if %b -> !qlx.logical_qubit {
        %z = qlx.apply #qlx.action<z>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
        cflow.yield %z : !qlx.logical_qubit
      } else {
        cflow.yield %q : !qlx.logical_qubit
      }
      cflow.yield %inner : !qlx.logical_qubit
    } else {
      %x = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
      cflow.yield %x : !qlx.logical_qubit
    }
    %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }

  // Loop state enters as a carry; a branch inside the body consumes the
  // carry once per path.
  // CHECK-LABEL: qlx.program @carried_loop
  qlx.program @carried_loop : (i1) -> i1 attributes {qlx.stage = "p0"} {
  ^bb0(%c: i1):
    %q0 = qlx.prepare "zero" : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" : !qlx.logical_qubit
    %r:2 = cflow.repeat 8
        iter(%a : !qlx.logical_qubit = %q0, %b : !qlx.logical_qubit = %q1) {
      %s = cflow.if %c -> !qlx.logical_qubit {
        %h = qlx.apply #qlx.action<h>(%a) : (!qlx.logical_qubit) -> !qlx.logical_qubit
        cflow.yield %h : !qlx.logical_qubit
      } else {
        cflow.yield %a : !qlx.logical_qubit
      }
      %x, %y = qlx.apply #qlx.action<cx>(%s, %b)
          : (!qlx.logical_qubit, !qlx.logical_qubit)
         -> (!qlx.logical_qubit, !qlx.logical_qubit)
      cflow.yield %x, %y : !qlx.logical_qubit, !qlx.logical_qubit
    }
    %m = qlx.measure <Z> %r#0 : !qlx.logical_qubit -> i1
    qlx.discard %r#1 : !qlx.logical_qubit
    qlx.return %m : i1
  }

  // A fresh qubit prepared and consumed inside the loop body is local to the
  // iteration.
  // CHECK-LABEL: qlx.program @loop_local
  qlx.program @loop_local : () -> () attributes {qlx.stage = "p0"} {
    cflow.repeat 3 iter() {
      %q = qlx.prepare "zero" : !qlx.logical_qubit
      qlx.discard %q : !qlx.logical_qubit
      cflow.yield
    }
    qlx.return
  }

  // Frame state is threaded linearly but may be dropped at the end.
  // CHECK-LABEL: qlx.program @dropped_frame
  qlx.program @dropped_frame : () -> i1 attributes {qlx.stage = "p0"} {
    %f = qlx.frame_init "pauli" : !qlx.logical_frame<"pauli">
    %q = qlx.prepare "zero" : !qlx.logical_qubit
    %m = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
    %f1 = qlx.frame_update %f from %m {update = "x"} : !qlx.logical_frame<"pauli"> -> !qlx.logical_frame<"pauli">
    qlx.return %m : i1
  }

  // Arguments flow to results.
  // CHECK-LABEL: qlx.objective_body @pass_through
  qlx.objective_body @pass_through : (!qlx.logical_qubit) -> !qlx.logical_qubit
      attributes {objective_kind = "action", qlx.stage = "p0"} {
  ^bb0(%q: !qlx.logical_qubit):
    %h = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    qlx.return %h : !qlx.logical_qubit
  }
}
