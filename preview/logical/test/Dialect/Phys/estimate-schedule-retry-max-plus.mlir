// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=slack result=slack_schedule' \
// RUN:   --phys-estimate-schedule='schedule=slack_schedule lower-tier=analytical result=slack_estimate' \
// RUN:   | FileCheck %s --check-prefix=SLACK
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=slack result=slack_schedule' \
// RUN:   --phys-estimate-schedule='schedule=slack_schedule lower-tier=analytical result=slack_estimate' \
// RUN:   | FileCheck %s --check-prefix=SLACK
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s \
// RUN:   --phys-schedule='graph=slack result=slack_schedule' \
// RUN:   --phys-estimate-schedule='schedule=slack_schedule lower-tier=analytical result=slack_estimate' \
// RUN:   2>&1 | FileCheck %s --check-prefix=SLACK-PROFILE
// RUN: qlx-opt %s --phys-schedule='graph=templated result=templated_schedule' \
// RUN:   --phys-estimate-schedule='schedule=templated_schedule lower-tier=analytical result=templated_estimate' \
// RUN:   | FileCheck %s --check-prefix=TEMPLATED
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=templated result=templated_schedule' \
// RUN:   --phys-estimate-schedule='schedule=templated_schedule lower-tier=analytical result=templated_estimate' \
// RUN:   | FileCheck %s --check-prefix=TEMPLATED
// RUN: qlx-opt %s --phys-schedule='graph=explicit result=explicit_schedule' \
// RUN:   --phys-estimate-schedule='schedule=explicit_schedule lower-tier=analytical result=explicit_estimate' \
// RUN:   | FileCheck %s --check-prefix=EXPLICIT
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=explicit result=explicit_schedule' \
// RUN:   --phys-estimate-schedule='schedule=explicit_schedule lower-tier=analytical result=explicit_estimate' \
// RUN:   | FileCheck %s --check-prefix=EXPLICIT
// RUN: qlx-opt %s \
// RUN:   --phys-schedule='graph=folded_parallel_template result=folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=folded_schedule lower-tier=analytical result=folded_estimate' \
// RUN:   | FileCheck %s --check-prefix=FOLDED
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=folded_parallel_template result=folded_schedule' \
// RUN:   --phys-estimate-schedule='schedule=folded_schedule lower-tier=analytical result=folded_estimate' \
// RUN:   | FileCheck %s --check-prefix=FOLDED
// RUN: qlx-opt %s \
// RUN:   --phys-schedule='graph=folded_partial_slack result=partial_schedule' \
// RUN:   --phys-estimate-schedule='schedule=partial_schedule lower-tier=analytical result=partial_estimate' \
// RUN:   | FileCheck %s --check-prefix=PARTIAL
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=folded_partial_slack result=partial_schedule' \
// RUN:   --phys-estimate-schedule='schedule=partial_schedule lower-tier=analytical result=partial_estimate' \
// RUN:   | FileCheck %s --check-prefix=PARTIAL
// RUN: qlx-opt %s \
// RUN:   --phys-schedule='graph=folded_resource_chain result=resource_chain_schedule' \
// RUN:   --phys-estimate-schedule='schedule=resource_chain_schedule lower-tier=analytical result=resource_chain_estimate' \
// RUN:   | FileCheck %s --check-prefix=RESOURCE-CHAIN
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=folded_resource_chain result=resource_chain_schedule' \
// RUN:   --phys-estimate-schedule='schedule=resource_chain_schedule lower-tier=analytical result=resource_chain_estimate' \
// RUN:   | FileCheck %s --check-prefix=RESOURCE-CHAIN

module attributes {qlx.profiles = ["p3"]} {
  lvm.domain @estimate_logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  fabric.code @estimate_code {
    distance = 1 : i64,
    metadata = {distance_method = "fixture", distance_provenance = @estimate_code_evidence},
    partitions = {data = 1 : i64}
  }
  fabric.code_profile @estimate_code_evidence {
    code = @estimate_code, distance_claim = 1 : i64,
    distance_status = "exact", evidence = ["schedule-estimate-fixture@1"]
  }
  fabric.machine @estimate_qec {
    fabric.region @compute {
      code = @estimate_code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
  }
  fabric.gadget @estimate_source() { fabric.return }
  fabric.gadget_spec @attempt_spec for @objective : () -> () {
    encodings = []
  }
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget @outer() {
    fabric.return
  }
  fabric.gadget_profile @attempt_profile for @attempt attributes {
    metadata = {
      success_probability = "0.5",
      success_probability_evidence = "analysis:max-plus-test"
    }
  } {}

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = ["measure_z"]
    }
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@qubits]
    }
  }
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["qubits"]}]
  }
  qlx.device @estimate_device {
    logical = @estimate_logical, qec = @estimate_qec, physical = @arch,
    logical_to_qec = @estimate_logical_to_qec,
    qec_to_physical = @estimate_qec_to_physical
  }
  qlx.estimate_result @static {
    assumptions = [], data = {}, device = @estimate_device,
    evidence = [@estimate_source], root = @estimate_source,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-counts/v1", tier = "static"
  }
  qlx.estimate_result @analytical {
    assumptions = [], data = {}, device = @estimate_device,
    evidence = [@estimate_source], lower_tier = @static,
    root = @estimate_source,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-estimate/v1", tier = "analytical"
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }

  // The left retry has two nanoseconds of maximum replay delay but ten
  // nanoseconds of slack to the baseline makespan.  Only the independent right
  // retry extends the 12 ns critical path: E[T]=13 and max(T)=14, not the
  // additive 14/16 produced by treating both singleton layers as serial.
  phys.graph @slack on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "slack.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %left:2 = "phys.call"(%q0) <{
      callee = @attempt, event_id = "slack.left.attempt",
      instance = "slack.left", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "slack.left.measure", record_id = "slack.left"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %left_ok = phys.condition %left#1 {event_id = "slack.left.decision"}
      : !phys.record<@attempt_bit> -> i1
    %left_result = phys.retry %left_ok carries (%left#0) {
      attempt = @attempt, attempt_event = "slack.left.attempt",
      decision_event = "slack.left.decision", event_id = "slack.left.retry",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:max-plus-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %late = phys.delay %q1 {
      duration_ns = 10.0 : f64, event_id = "slack.delay"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %right:2 = "phys.call"(%late) <{
      callee = @attempt, event_id = "slack.right.attempt",
      instance = "slack.right", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q1>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "slack.right.measure", record_id = "slack.right"
      } : (!phys.state<@q1>) ->
          (!phys.state<@q1>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q1>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q1>) ->
        (!phys.state<@q1>, !phys.record<@attempt_bit>)
    %right_ok = phys.condition %right#1 {event_id = "slack.right.decision"}
      : !phys.record<@attempt_bit> -> i1
    %right_result = phys.retry %right_ok carries (%right#0) {
      attempt = @attempt, attempt_event = "slack.right.attempt",
      decision_event = "slack.right.decision", event_id = "slack.right.retry",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:max-plus-test"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %left_result, %right_result
      : !phys.state<@q0>, !phys.state<@q1>
  }

  phys.graph @templated on @arch : () -> !phys.state<@q0> attributes {
    source_protocol = @estimate_source
  } {
    %q = phys.acquire [@q0] {event_id = "templated.acquire"}
      : !phys.state<@q0>
    %first = "phys.call"(%q) <{
      callee = @outer, event_id = "templated.outer0",
      instance = "templated.outer.0"
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%current) <{
        callee = @attempt, event_id = "templated.attempt",
        instance = "templated.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "templated.measure", record_id = "templated"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "templated.decision"
      } : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "templated.attempt",
        decision_event = "templated.decision", event_id = "templated.retry",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:max-plus-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %retried : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %second = phys.call_template %first {
      callee = @outer, event_id = "templated.outer1",
      instance = "templated.outer.1", template_event = "templated.outer0"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %second : !phys.state<@q0>
  }

  phys.graph @explicit on @arch : () -> !phys.state<@q0> attributes {
    source_protocol = @estimate_source
  } {
    %q = phys.acquire [@q0] {event_id = "explicit.acquire"}
      : !phys.state<@q0>
    %first:2 = "phys.call"(%q) <{
      callee = @attempt, event_id = "explicit.attempt0",
      instance = "explicit.attempt.0", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "explicit.measure0", record_id = "explicit.0"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %accepted0 = phys.condition %first#1 {event_id = "explicit.decision0"}
      : !phys.record<@attempt_bit> -> i1
    %retried0 = phys.retry %accepted0 carries (%first#0) {
      attempt = @attempt, attempt_event = "explicit.attempt0",
      decision_event = "explicit.decision0", event_id = "explicit.retry0",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:max-plus-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %second:2 = "phys.call"(%retried0) <{
      callee = @attempt, event_id = "explicit.attempt1",
      instance = "explicit.attempt.1", profile = @attempt_profile
    }> ({
    ^bb0(%current: !phys.state<@q0>):
      %measured:2 = phys.measure @measure_z(%current) {
        event_id = "explicit.measure1", record_id = "explicit.1"
      } : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      phys.yield %measured#0, %measured#1
        : !phys.state<@q0>, !phys.record<@attempt_bit>
    }) : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@attempt_bit>)
    %accepted1 = phys.condition %second#1 {event_id = "explicit.decision1"}
      : !phys.record<@attempt_bit> -> i1
    %retried1 = phys.retry %accepted1 carries (%second#0) {
      attempt = @attempt, attempt_event = "explicit.attempt1",
      decision_event = "explicit.decision1", event_id = "explicit.retry1",
      max_attempts = 2 : i64, profile = @attempt_profile,
      success_probability = 5.000000e-01 : f64,
      success_probability_source = @attempt_profile,
      success_probability_evidence = "analysis:max-plus-test"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %retried1 : !phys.state<@q0>
  }

  // One compact repeat owns both the canonical RUS call and a parallel
  // call-template occurrence of it.  The two retry variables have distinct
  // invocation identities, but execute in the same three folded iterations.
  // Their per-iteration completion time is therefore the maximum of two
  // truncated geometric variables, not two serial delays and not two
  // unrelated three-attempt sums.
  phys.graph @folded_parallel_template on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "folded.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %result:2 = "cflow.repeat"(%q0, %q1) <{
      count = 3 : i64, event_id = "folded.repeat"
    }> ({
    ^bb0(%left: !phys.state<@q0>, %right: !phys.state<@q1>):
      %canonical = "phys.call"(%left) <{
        callee = @outer, event_id = "folded.outer0",
        instance = "folded.outer.0"
      }> ({
      ^bb0(%current: !phys.state<@q0>):
        %attempted:2 = "phys.call"(%current) <{
          callee = @attempt, event_id = "folded.attempt",
          instance = "folded.attempt", profile = @attempt_profile
        }> ({
        ^bb0(%attempt_state: !phys.state<@q0>):
          %measured:2 = phys.measure @measure_z(%attempt_state) {
            event_id = "folded.measure", record_id = "folded"
          } : (!phys.state<@q0>) ->
              (!phys.state<@q0>, !phys.record<@attempt_bit>)
          phys.yield %measured#0, %measured#1
            : !phys.state<@q0>, !phys.record<@attempt_bit>
        }) : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        %accepted = phys.condition %attempted#1 {
          event_id = "folded.decision"
        } : !phys.record<@attempt_bit> -> i1
        %retried = phys.retry %accepted carries (%attempted#0) {
          attempt = @attempt, attempt_event = "folded.attempt",
          decision_event = "folded.decision", event_id = "folded.retry",
          max_attempts = 2 : i64, profile = @attempt_profile,
          success_probability = 5.000000e-01 : f64,
          success_probability_source = @attempt_profile,
          success_probability_evidence = "analysis:max-plus-test"
        } : (!phys.state<@q0>) -> !phys.state<@q0>
        phys.yield %retried : !phys.state<@q0>
      }) : (!phys.state<@q0>) -> !phys.state<@q0>
      %clone = phys.call_template %right {
        callee = @outer, event_id = "folded.outer1",
        instance = "folded.outer.1",
        state_aliases = [{alias = @q1, template = @q0}],
        template_event = "folded.outer0"
      } : (!phys.state<@q1>) -> !phys.state<@q1>
      cflow.yield %canonical, %clone
        : !phys.state<@q0>, !phys.state<@q1>
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    phys.return %result#0, %result#1
      : !phys.state<@q0>, !phys.state<@q1>
  }

  // The retry can add two nanoseconds, but an independent three-nanosecond
  // branch leaves one nanosecond of slack after its two-nanosecond first
  // attempt.  Each of three dynamic iterations therefore contributes
  // max(0, 2B - 1), B~Bernoulli(0.5): expected 0.5 ns, maximum 1 ns.
  phys.graph @folded_partial_slack on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "partial.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %result:2 = "cflow.repeat"(%q0, %q1) <{
      count = 3 : i64, event_id = "partial.repeat"
    }> ({
    ^bb0(%active: !phys.state<@q0>, %independent: !phys.state<@q1>):
      %attempted:2 = "phys.call"(%active) <{
        callee = @attempt, event_id = "partial.attempt",
        instance = "partial.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "partial.measure", record_id = "partial"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "partial.decision"
      } : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "partial.attempt",
        decision_event = "partial.decision", event_id = "partial.retry",
        max_attempts = 2 : i64, profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:max-plus-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      %slack = phys.delay %independent {
        duration_ns = 3.0 : f64, event_id = "partial.slack"
      } : (!phys.state<@q1>) -> !phys.state<@q1>
      cflow.yield %retried, %slack
        : !phys.state<@q0>, !phys.state<@q1>
    }) : (!phys.state<@q0>, !phys.state<@q1>) ->
        (!phys.state<@q0>, !phys.state<@q1>)
    phys.return %result#0, %result#1
      : !phys.state<@q0>, !phys.state<@q1>
  }

  // The three-nanosecond tail is later in wall-clock time, but it consumes the
  // retry's same q0 state.  Replay delay therefore propagates through the tail
  // with zero causal slack.  A timestamp-only fold calculation incorrectly
  // hides the two-nanosecond replay in that tail; the dependency-aware result
  // adds one expected and two maximum nanoseconds in each of three iterations.
  phys.graph @folded_resource_chain on @arch : () -> !phys.state<@q0>
      attributes {source_protocol = @estimate_source} {
    %q0 = phys.acquire [@q0] {event_id = "resource_chain.acquire"}
      : !phys.state<@q0>
    %result = "cflow.repeat"(%q0) <{
      count = 3 : i64, event_id = "resource_chain.repeat"
    }> ({
    ^bb0(%active: !phys.state<@q0>):
      %attempted:2 = "phys.call"(%active) <{
        callee = @attempt, event_id = "resource_chain.attempt",
        instance = "resource_chain.attempt", profile = @attempt_profile
      }> ({
      ^bb0(%attempt_state: !phys.state<@q0>):
        %measured:2 = phys.measure @measure_z(%attempt_state) {
          event_id = "resource_chain.measure", record_id = "resource_chain"
        } : (!phys.state<@q0>) ->
            (!phys.state<@q0>, !phys.record<@attempt_bit>)
        phys.yield %measured#0, %measured#1
          : !phys.state<@q0>, !phys.record<@attempt_bit>
      }) : (!phys.state<@q0>) ->
          (!phys.state<@q0>, !phys.record<@attempt_bit>)
      %accepted = phys.condition %attempted#1 {
        event_id = "resource_chain.decision"
      } : !phys.record<@attempt_bit> -> i1
      %retried = phys.retry %accepted carries (%attempted#0) {
        attempt = @attempt, attempt_event = "resource_chain.attempt",
        decision_event = "resource_chain.decision",
        event_id = "resource_chain.retry", max_attempts = 2 : i64,
        profile = @attempt_profile,
        success_probability = 5.000000e-01 : f64,
        success_probability_source = @attempt_profile,
        success_probability_evidence = "analysis:max-plus-test"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      %tail = phys.delay %retried {
        duration_ns = 3.0 : f64, event_id = "resource_chain.tail"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      cflow.yield %tail : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %result : !phys.state<@q0>
  }
}

// SLACK: qlx.estimate_result @slack_estimate
// SLACK-SAME: "scheduled_macro durations are deterministic mean-output slots;
// SLACK-SAME: maximum_makespan_ns is conditional on those slots, not a physical factory worst case"
// SLACK-SAME: exhaustion_probability = 4.375000e-01 : f64
// SLACK-SAME: expected_makespan_ns = 1.300000e+01 : f64
// SLACK-SAME: makespan_ns = 1.200000e+01 : f64
// SLACK-SAME: maximum_makespan_ns = 1.400000e+01 : f64

// SLACK-PROFILE: phys-estimate-schedule retry-causality singleton-groups=2 parallel-groups=0 hidden-layers=1 contributing-layers=1 owner-propagation-rows=0 dependency-edges=0 controlled-events=0

// TEMPLATED: qlx.estimate_result @templated_estimate
// TEMPLATED-SAME: exhaustion_probability = 4.375000e-01 : f64
// TEMPLATED-SAME: expected_active_physical_qubit_time_ns = 3.000000e+00 : f64
// TEMPLATED-SAME: expected_active_resource_time_ns = 3.000000e+00 : f64
// TEMPLATED-SAME: expected_makespan_ns = 6.000000e+00 : f64
// TEMPLATED-SAME: makespan_ns = 4.000000e+00 : f64
// TEMPLATED-SAME: maximum_active_physical_qubit_time_ns = 4.000000e+00 : f64
// TEMPLATED-SAME: maximum_active_resource_time_ns = 4.000000e+00 : f64
// TEMPLATED-SAME: maximum_makespan_ns = 8.000000e+00 : f64

// EXPLICIT: qlx.estimate_result @explicit_estimate
// EXPLICIT-SAME: exhaustion_probability = 4.375000e-01 : f64
// EXPLICIT-SAME: expected_active_physical_qubit_time_ns = 3.000000e+00 : f64
// EXPLICIT-SAME: expected_active_resource_time_ns = 3.000000e+00 : f64
// EXPLICIT-SAME: expected_makespan_ns = 6.000000e+00 : f64
// EXPLICIT-SAME: makespan_ns = 4.000000e+00 : f64
// EXPLICIT-SAME: maximum_active_physical_qubit_time_ns = 4.000000e+00 : f64
// EXPLICIT-SAME: maximum_active_resource_time_ns = 4.000000e+00 : f64
// EXPLICIT-SAME: maximum_makespan_ns = 8.000000e+00 : f64

// FOLDED: qlx.estimate_result @folded_estimate
// FOLDED-SAME: exhaustion_probability = 0.822021484375 : f64
// FOLDED-SAME: expected_makespan_ns = 1.050000e+01 : f64
// FOLDED-SAME: makespan_ns = 6.000000e+00 : f64
// FOLDED-SAME: maximum_makespan_ns = 1.200000e+01 : f64

// PARTIAL: qlx.estimate_result @partial_estimate
// PARTIAL-SAME: exhaustion_probability = 5.781250e-01 : f64
// PARTIAL-SAME: expected_makespan_ns = 1.050000e+01 : f64
// PARTIAL-SAME: makespan_ns = 9.000000e+00 : f64
// PARTIAL-SAME: maximum_makespan_ns = 1.200000e+01 : f64

// RESOURCE-CHAIN: qlx.estimate_result @resource_chain_estimate
// RESOURCE-CHAIN-SAME: expected_makespan_ns = 1.800000e+01 : f64
// RESOURCE-CHAIN-SAME: makespan_ns = 1.500000e+01 : f64
// RESOURCE-CHAIN-SAME: maximum_makespan_ns = 2.100000e+01 : f64
