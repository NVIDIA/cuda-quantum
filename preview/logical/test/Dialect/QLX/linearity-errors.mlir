// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s

// Linear single ownership in P0. The qlx.program and qlx.objective_body
// verifiers reject each case below; plain SSA verification accepts all of
// them.

/// Two gates applied to the same logical qubit.
qlx.program @double_apply : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-note@+1 {{earlier consume here}}
  %a = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  // expected-error@+1 {{linear value %0 : !qlx.logical_qubit (result of qlx.prepare) is consumed more than once: by qlx.apply and again by qlx.apply}}
  %b = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  %m = qlx.measure <Z> %a : !qlx.logical_qubit -> i1
  qlx.discard %b : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// One op that consumes the same value through two operands.
qlx.program @same_op_twice : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-note@+2 {{earlier consume here}}
  // expected-error@+1 {{is consumed more than once: by qlx.apply and again by qlx.apply}}
  %a, %b = qlx.apply #qlx.action<cx>(%q, %q)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit)
  %m = qlx.measure <Z> %a : !qlx.logical_qubit -> i1
  qlx.discard %b : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// Measurement is destructive: the measured value has no successor.
qlx.program @use_after_measure : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-note@+1 {{earlier consume here}}
  %m = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
  // expected-error@+1 {{is consumed more than once: by qlx.measure and again by qlx.apply}}
  %h = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  qlx.discard %h : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// A second measurement of the same value.
qlx.program @double_measure : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-note@+1 {{earlier consume here}}
  %a = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
  // expected-error@+1 {{is consumed more than once: by qlx.measure and again by qlx.measure}}
  %b = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
  %x = qlx.xor %a, %b : i1
  qlx.return %x : i1
}

// -----

/// A prepared qubit that is never consumed leaks.
qlx.program @leaked_prepare : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-error@+1 {{linear value %1 : !qlx.logical_qubit (result of qlx.prepare) is never consumed}}
  %q1 = qlx.prepare "zero" : !qlx.logical_qubit
  %m = qlx.measure <Z> %q0 : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A gate's successor value leaks.
qlx.program @leaked_successor : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-error@+1 {{linear value %2#1 : !qlx.logical_qubit (result of qlx.apply) is never consumed}}
  %c, %t = qlx.apply #qlx.action<cx>(%q0, %q1)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
     -> (!qlx.logical_qubit, !qlx.logical_qubit)
  %m = qlx.measure <Z> %c : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// An entry-block argument is an owner as well.
// expected-error@+1 {{linear value %arg0 : !qlx.logical_qubit (block argument 0 of qlx.program) is never consumed}}
qlx.program @leaked_argument : (!qlx.logical_qubit) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%q: !qlx.logical_qubit):
  %false = arith.constant false
  qlx.return %false : i1
}

// -----

/// A value captured into a cflow.repeat body is consumed once per iteration.
qlx.program @repeat_capture : () -> i1 attributes {qlx.stage = "p0"} {
  %q0 = qlx.prepare "zero" : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" : !qlx.logical_qubit
  %r = cflow.repeat 4 iter(%arg : !qlx.logical_qubit = %q0) {
    // expected-error@+1 {{linear value %1 : !qlx.logical_qubit (result of qlx.prepare) defined outside cflow.repeat is consumed inside its body by qlx.apply; loop-carried linear state must be threaded as an explicit init/carry}}
    %c0, %c1 = qlx.apply #qlx.action<cx>(%arg, %q1)
        : (!qlx.logical_qubit, !qlx.logical_qubit)
       -> (!qlx.logical_qubit, !qlx.logical_qubit)
    qlx.discard %c1 : !qlx.logical_qubit
    cflow.yield %c0 : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Capturing the loop init alongside its carry is also a loop capture.
qlx.program @repeat_capture_of_init : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  %r = cflow.repeat 4 iter(%arg : !qlx.logical_qubit = %q) {
    // expected-error@+1 {{defined outside cflow.repeat is consumed inside its body by qlx.apply}}
    %t = qlx.apply #qlx.action<t>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    qlx.discard %arg : !qlx.logical_qubit
    cflow.yield %t : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A capture nested under an if inside the loop is still a loop capture.
qlx.program @repeat_capture_under_if : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%c: i1):
  %q0 = qlx.prepare "zero" : !qlx.logical_qubit
  %q1 = qlx.prepare "zero" : !qlx.logical_qubit
  %r = cflow.repeat 2 iter(%arg : !qlx.logical_qubit = %q0) {
    cflow.if %c {
      // expected-error@+1 {{defined outside cflow.repeat is consumed inside its body by qlx.discard}}
      qlx.discard %q1 : !qlx.logical_qubit
      cflow.yield
    } else {
      qlx.discard %q1 : !qlx.logical_qubit
      cflow.yield
    }
    cflow.yield %arg : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A cflow.while body consuming a captured value.
qlx.program @while_capture : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%go: i1):
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  %0 = "cflow.while"(%go) <{max_iterations = 8 : i64}> ({
  ^bb0(%current: i1):
    "cflow.while_condition"(%current, %current) : (i1, i1) -> ()
  }, {
  ^bb0(%current: i1):
    // expected-error@+1 {{defined outside cflow.while is consumed inside its body by qlx.discard}}
    qlx.discard %q : !qlx.logical_qubit
    cflow.yield %current : i1
  }) : (i1) -> i1
  qlx.return %0 : i1
}

// -----

/// A loop carry consumed twice in one iteration.
qlx.program @repeat_carry_double : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  %r = cflow.repeat 4 iter(%arg : !qlx.logical_qubit = %q) {
    // expected-note@+1 {{earlier consume here}}
    %h = qlx.apply #qlx.action<h>(%arg) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    qlx.discard %h : !qlx.logical_qubit
    // expected-error@+1 {{linear value %arg0 : !qlx.logical_qubit (block argument 0 of cflow.repeat) is consumed more than once: by qlx.apply and again by cflow.yield}}
    cflow.yield %arg : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A loop carry dropped inside the body leaks the iteration's owner.
qlx.program @repeat_carry_leak : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-error@+1 {{linear value %arg0 : !qlx.logical_qubit (block argument 0 of cflow.repeat) is never consumed}}
  %r = cflow.repeat 4 iter(%arg : !qlx.logical_qubit = %q) {
    %fresh = qlx.prepare "zero" : !qlx.logical_qubit
    cflow.yield %fresh : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A loop init consumed again after the loop.
qlx.program @repeat_init_reused : () -> i1 attributes {qlx.stage = "p0"} {
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-note@+1 {{earlier consume here}}
  %r = cflow.repeat 4 iter(%arg : !qlx.logical_qubit = %q) {
    cflow.yield %arg : !qlx.logical_qubit
  }
  qlx.discard %r : !qlx.logical_qubit
  // expected-error@+1 {{is consumed more than once: by cflow.repeat and again by qlx.measure}}
  %m = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// Consuming a captured value in one branch only leaks it on the other path.
qlx.program @if_one_branch : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%c: i1):
  // expected-note@+1 {{value defined here}}
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-error@+1 {{linear value %0 : !qlx.logical_qubit (result of qlx.prepare) is consumed in some but not all regions of cflow.if; it must be consumed on every path}}
  %r = cflow.if %c -> !qlx.logical_qubit {
    %x = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %x : !qlx.logical_qubit
  } else {
    %fresh = qlx.prepare "zero" : !qlx.logical_qubit
    cflow.yield %fresh : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// A branch consume plus a consume after the if is a double use.
qlx.program @if_branch_and_after : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%c: i1):
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  %r = cflow.if %c -> !qlx.logical_qubit {
    // expected-note@+1 {{earlier consume here}}
    %x = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    cflow.yield %x : !qlx.logical_qubit
  } else {
    cflow.yield %q : !qlx.logical_qubit
  }
  // expected-error@+1 {{is consumed more than once: by qlx.apply and again by qlx.measure}}
  %m = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
  qlx.discard %r : !qlx.logical_qubit
  qlx.return %m : i1
}

// -----

/// Two consumes within the same branch.
qlx.program @if_double_in_branch : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%c: i1):
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  %r = cflow.if %c -> !qlx.logical_qubit {
    // expected-note@+1 {{earlier consume here}}
    %x = qlx.apply #qlx.action<x>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
    qlx.discard %x : !qlx.logical_qubit
    // expected-error@+1 {{is consumed more than once: by qlx.apply and again by cflow.yield}}
    cflow.yield %q : !qlx.logical_qubit
  } else {
    cflow.yield %q : !qlx.logical_qubit
  }
  %m = qlx.measure <Z> %r : !qlx.logical_qubit -> i1
  qlx.return %m : i1
}

// -----

/// An if result that is never consumed leaks.
qlx.program @if_result_leak : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%c: i1):
  %q = qlx.prepare "zero" : !qlx.logical_qubit
  // expected-error@+1 {{linear value %1 : !qlx.logical_qubit (result of cflow.if) is never consumed}}
  %r = cflow.if %c -> !qlx.logical_qubit {
    cflow.yield %q : !qlx.logical_qubit
  } else {
    cflow.yield %q : !qlx.logical_qubit
  }
  %false = arith.constant false
  qlx.return %false : i1
}

// -----

/// Objective bodies follow the same rule.
qlx.objective_body @double_use_objective : (!qlx.logical_qubit) -> !qlx.logical_qubit
    attributes {objective_kind = "action", qlx.stage = "p0"} {
^bb0(%q: !qlx.logical_qubit):
  // expected-note@+1 {{earlier consume here}}
  %a = qlx.apply #qlx.action<h>(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  qlx.discard %a : !qlx.logical_qubit
  // expected-error@+1 {{linear value %arg0 : !qlx.logical_qubit (block argument 0 of qlx.objective_body) is consumed more than once: by qlx.apply and again by qlx.return}}
  qlx.return %q : !qlx.logical_qubit
}

// -----

/// A logical resource is linear too.
qlx.program @resource_leak : () -> i1 attributes {qlx.stage = "p0"} {
  // expected-error@+1 {{linear value %0 : !event.handle<!qlx.logical_resource<"t_state">, "linear"> (result of qlx.resource_request) is never consumed}}
  %e = qlx.resource_request "t_state" : !event.handle<!qlx.logical_resource<"t_state">, "linear">
  %false = arith.constant false
  qlx.return %false : i1
}
