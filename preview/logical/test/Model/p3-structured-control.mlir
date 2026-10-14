// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = ["measure_z", "x"]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @feedback on @arch : () -> !phys.state<@q0> {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %1:2 = phys.measure @measure_z(%0) {
      event_id = "m0", record_id = "m"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    %2 = phys.condition %1#1 {event_id = "condition0"}
      : !phys.record<@bit> -> i1
    %3 = "cflow.if"(%2) <{event_id = "if0"}> ({
      %4 = phys.apply @x(%1#0) {event_id = "x0"}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %4 : !phys.state<@q0>
    }, {
      cflow.yield %1#0 : !phys.state<@q0>
    }) : (i1) -> !phys.state<@q0>
    %4:2 = "phys.call"(%3) <{
      callee = @attempt, event_id = "attempt0",
      instance = "attempt.instance0", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %attempted:2 = phys.measure @measure_z(%current) {
        event_id = "attempt_measure0", record_id = "attempt_result"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %attempted#0, %attempted#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %5 = phys.condition %4#1 {event_id = "decision0"}
      : !phys.record<@attempt_bit> -> i1
    %6 = phys.retry %5 carries (%4#0) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", event_id = "retry0",
      max_attempts = 3 : i64, profile = @attempt_profile
    }
      : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %6 : !phys.state<@q0>
  }

  func.func @dynamic_loop(%go: i1, %q: !phys.state<@q0>)
      -> (!phys.state<@q0>, i1) {
    %0:2 = "cflow.while"(%q, %go) <{max_iterations = 4 : i64,
                                      event_id = "while0"}> ({
    ^bb0(%current: !phys.state<@q0>, %predicate: i1):
      "cflow.while_condition"(%predicate, %current, %predicate)
        : (i1, !phys.state<@q0>, i1) -> ()
    }, {
    ^bb0(%current: !phys.state<@q0>, %predicate: i1):
      cflow.yield %current, %predicate : !phys.state<@q0>, i1
    }) : (!phys.state<@q0>, i1) -> (!phys.state<@q0>, i1)
    return %0#0, %0#1 : !phys.state<@q0>, i1
  }

  func.func @folded_repeat(%q: !phys.state<@q0>) -> !phys.state<@q0> {
    %0 = "cflow.repeat"(%q) <{count = 1000000 : i64,
                                event_id = "repeat0"}> ({
    ^bb0(%current: !phys.state<@q0>):
      %1 = phys.apply @x(%current) {event_id = "repeat_x"}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %1 : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    return %0 : !phys.state<@q0>
  }
}

// CHECK: cflow.if
// CHECK: cflow.yield
// CHECK: "phys.call"
// CHECK-SAME: callee = @attempt
// CHECK-SAME: profile = @attempt_profile
// CHECK: phys.condition
// CHECK: phys.retry
// CHECK-SAME: attempt = @attempt
// CHECK-SAME: attempt_event = "attempt0"
// CHECK-SAME: decision_event = "decision0"
// CHECK-SAME: max_attempts = 3 : i64
// CHECK-SAME: profile = @attempt_profile
// CHECK: cflow.while(
// CHECK-SAME: max_iterations = 4
// CHECK: cflow.while_condition
// CHECK: cflow.repeat 1000000
// CHECK: phys.apply @x
