// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

// `cflow.repeat`'s `count` bound is the shared control-flow dialect's own
// verifier contract (see test/Dialect/Cflow/error-repeat.mlir); it is not
// re-tested here.

module {
  func.func @bad_condition(%q: !phys.state<@q>) {
    // expected-error @+1 {{source must be i1 or a phys.record}}
    %0 = phys.condition %q : !phys.state<@q> -> i1
    return
  }
}

// `cflow.while`'s `max_iterations` bound is the shared control-flow
// dialect's own verifier contract (see test/Dialect/Cflow/error-while.mlir);
// it is not re-tested here.

module {
  func.func @bad_retry(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{max_attempts must be positive}}
    %0 = phys.retry %ok carries (%q) {
      attempt = @attempt,
      attempt_event = "attempt0",
      decision_event = "decision0",
      max_attempts = 0 : i64,
      profile = @profile
    }
      : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  fabric.gadget_profile @wrong_profile for @attempt {}

  func.func @bad_retry_contract(%q: !phys.state<@q>) {
    %0:2 = "phys.call"(%q) <{
      callee = @attempt,
      event_id = "attempt0",
      instance = "attempt.instance0",
      profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %accepted = arith.constant true
      phys.yield %current, %accepted : !phys.state<@q>, i1
    }) : (!phys.state<@q>) -> (!phys.state<@q>, i1)
    %decision = phys.xor %0#1, %0#1 {event_id = "decision0"} : i1
    // expected-error @+1 {{attempt_event profile must equal the selected profile}}
    %1 = phys.retry %decision carries (%0#0) {
      attempt = @attempt,
      attempt_event = "attempt0",
      commit_point = "before_output",
      decision_event = "decision0",
      max_attempts = 2 : i64,
      profile = @wrong_profile
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }

  func.func @bad_retry_omitted_state(
      %a: !phys.state<@q0>, %b: !phys.state<@q1>) {
    %0:3 = "phys.call"(%a, %b) <{
      callee = @attempt,
      event_id = "attempt1",
      instance = "attempt.instance1",
      profile = @attempt_profile
    }> ({
    ^bb0(%current_a: !phys.state<@q0>, %current_b: !phys.state<@q1>):
      %accepted = arith.constant true
      phys.yield %current_a, %current_b, %accepted
        : !phys.state<@q0>, !phys.state<@q1>, i1
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>, i1)
    %decision = phys.xor %0#2, %0#2 {event_id = "decision1"} : i1
    // expected-error @+1 {{every attempt physical-state result must be carried exactly once}}
    %1 = phys.retry %decision carries (%0#0) {
      attempt = @attempt,
      attempt_event = "attempt1",
      commit_point = "before_output",
      decision_event = "decision1",
      max_attempts = 2 : i64,
      profile = @attempt_profile
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    return
  }
}

// -----

module {
  func.func @missing_retry_attempt(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{requires attribute 'attempt'}}
    %0 = phys.retry %ok carries (%q) {
      attempt_event = "attempt0", decision_event = "decision0",
      max_attempts = 3 : i64, profile = @profile
    }
      : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @bogus_profile for @attempt attributes {
    metadata = {
      success_probability = "0.5",
      success_probability_evidence = "bogus"
    }
  } {}
  func.func @unsupported_retry_evidence(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{success_probability evidence must be synthesis:sha256:... or analysis:...}}
    %0 = phys.retry %ok carries (%q) {
      attempt = @attempt,
      attempt_event = "attempt0",
      decision_event = "decision0",
      max_attempts = 3 : i64,
      profile = @bogus_profile,
      success_probability = 0.5 : f64,
      success_probability_source = @bogus_profile,
      success_probability_evidence = "bogus"
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @synthesis_profile for @attempt attributes {
    metadata = {
      success_probability = "0.5",
      success_probability_evidence = "synthesis:sha256:0000000000000000000000000000000000000000000000000000000000000000",
      synthesis_sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
    }
  } {}
  func.func @mismatched_retry_digest(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{synthesis probability evidence must bind the attempt digest}}
    %0 = phys.retry %ok carries (%q) {
      attempt = @attempt,
      attempt_event = "attempt0",
      decision_event = "decision0",
      max_attempts = 3 : i64,
      profile = @synthesis_profile,
      success_probability = 0.5 : f64,
      success_probability_source = @synthesis_profile,
      success_probability_evidence = "synthesis:sha256:0000000000000000000000000000000000000000000000000000000000000000"
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget_spec @other_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget @other() { fabric.return }
    {realization_boundary = {}, spec = @other_spec}
  fabric.gadget_profile @analysis for @other {}
  func.func @unrelated_retry_profile(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{profile must analyze the retry attempt}}
    %0 = phys.retry %ok carries (%q) {
      attempt = @attempt,
      attempt_event = "attempt0",
      decision_event = "decision0",
      max_attempts = 3 : i64,
      profile = @analysis
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  func.func @bad_retry_probability(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{success_probability must be finite and lie in (0, 1]}}
    %0 = phys.retry %ok carries (%q) {
      attempt = @attempt,
      attempt_event = "attempt0",
      decision_event = "decision0",
      max_attempts = 3 : i64,
      profile = @attempt_profile,
      success_probability = 1.1 : f64
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  func.func @uncertified_retry_probability(%ok: i1, %q: !phys.state<@q>) {
    // expected-error @+1 {{success_probability requires a source and nonempty evidence}}
    %0 = phys.retry %ok carries (%q) {
      attempt = @attempt,
      attempt_event = "attempt0",
      decision_event = "decision0",
      max_attempts = 3 : i64,
      profile = @attempt_profile,
      success_probability = 0.5 : f64
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

// `cflow.if`'s branch yield-type/count agreement is the shared
// control-flow dialect's own verifier contract (see
// test/Dialect/Cflow/error-if.mlir); it is not re-tested here.

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget_spec @other_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget @other() { fabric.return }
    {realization_boundary = {}, spec = @other_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  fabric.gadget_profile @other_profile for @other {}
  func.func @retry_names_another_attempt(%q: !phys.state<@q>) {
    %attempted:2 = "phys.call"(%q) <{
      callee = @attempt, event_id = "attempt0",
      instance = "retry_names_another_attempt.call0",
      profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %ok = arith.constant true
      phys.yield %current, %ok : !phys.state<@q>, i1
    }) : (!phys.state<@q>) -> (!phys.state<@q>, i1)
    %decision = phys.xor %attempted#1, %attempted#1 {
      event_id = "decision0"
    } : i1
    // expected-error @+1 {{attempt_event callee must equal the selected attempt}}
    %0 = phys.retry %decision carries (%attempted#0) {
      attempt = @other, attempt_event = "attempt0",
      decision_event = "decision0", max_attempts = 3 : i64,
      profile = @other_profile
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  func.func @retry_carries_another_call(%q: !phys.state<@q>) {
    %first:2 = "phys.call"(%q) <{
      callee = @attempt, event_id = "attempt0",
      instance = "retry_carries_another_call.call0",
      profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %ok = arith.constant true
      phys.yield %current, %ok : !phys.state<@q>, i1
    }) : (!phys.state<@q>) -> (!phys.state<@q>, i1)
    %second:2 = "phys.call"(%first#0) <{
      callee = @attempt, event_id = "attempt1",
      instance = "retry_carries_another_call.call1",
      profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %ok = arith.constant true
      phys.yield %current, %ok : !phys.state<@q>, i1
    }) : (!phys.state<@q>) -> (!phys.state<@q>, i1)
    %decision = phys.xor %first#1, %first#1 {
      event_id = "decision0"
    } : i1
    // expected-error @+1 {{every carried physical state must originate from attempt_event}}
    %0 = phys.retry %decision carries (%second#0) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", max_attempts = 3 : i64,
      profile = @attempt_profile
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  func.func @retry_uses_unrelated_predicate(%ok: i1, %q: !phys.state<@q>) {
    %attempted:2 = "phys.call"(%q) <{
      callee = @attempt, event_id = "attempt0",
      instance = "retry_uses_unrelated_predicate.call0",
      profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %attempt_ok = arith.constant true
      phys.yield %current, %attempt_ok : !phys.state<@q>, i1
    }) : (!phys.state<@q>) -> (!phys.state<@q>, i1)
    %decision = phys.xor %ok, %ok {event_id = "decision0"} : i1
    // expected-error @+1 {{decision_event must be causally derived from attempt_event results}}
    %0 = phys.retry %decision carries (%attempted#0) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", max_attempts = 3 : i64,
      profile = @attempt_profile
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() { fabric.return }
    {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  func.func @retry_condition_cannot_transform_direct_i1(%q: !phys.state<@q>) {
    %attempted:2 = "phys.call"(%q) <{
      callee = @attempt, event_id = "attempt0",
      instance = "retry_condition_cannot_transform_direct_i1.call0",
      profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %ok = arith.constant true
      phys.yield %current, %ok : !phys.state<@q>, i1
    }) : (!phys.state<@q>) -> (!phys.state<@q>, i1)
    %decision = phys.condition %attempted#1 {event_id = "decision0"} : i1 -> i1
    // Identity conditioning is the canonical typed decision event for a direct
    // i1 attempt result; only inversion (`expected = false`) is forbidden.
    %0 = phys.retry %decision carries (%attempted#0) {
      attempt = @attempt, attempt_event = "attempt0",
      decision_event = "decision0", max_attempts = 3 : i64,
      profile = @attempt_profile
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}
