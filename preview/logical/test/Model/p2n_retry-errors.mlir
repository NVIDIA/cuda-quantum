// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.protocol @attempt : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
  ^bb0(%arg0: !fabric.patch<@c>):
    %ok = arith.constant true
    fabric.protocol_return %arg0, %ok : !fabric.patch<@c>, i1
  }
  fabric.protocol @claimed : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
  ^bb0(%arg0: !fabric.patch<@c>):
    %ok = arith.constant true
    fabric.protocol_return %arg0, %ok : !fabric.patch<@c>, i1
  }
  fabric.gadget_spec @evidenced_spec for @objective
      : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
    encodings = [], record_schema = ["accepted.outcome"],
    outcome_map = {
      records = ["accepted.outcome"], rows = dense<1> : tensor<1x1xi1>,
      constants = array<i64: 0>, input_syndromes = [[]], roles = [["success"]]
    }
  }
  fabric.gadget @evidenced_gadget(%arg0: !fabric.patch<@c>)
      -> (!fabric.patch<@c>, i1) {
    %next, %ok = fabric.measure_product %arg0 {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "accepted"
    } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    fabric.return %next, %ok : !fabric.patch<@c>, i1
  } {realization_boundary = {}, spec = @evidenced_spec}
  fabric.gadget_profile @evidenced_profile for @evidenced_gadget {
    fabric.success {records = ["evidenced_gadget.accepted.outcome"]}
  }
  fabric.protocol @evidenced_attempt
      : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1) attributes {
    predicate_gadget = @evidenced_gadget,
    predicate_profile = @evidenced_profile,
    predicate_result = 1 : i64,
    metadata = {
      success_probability = "0.5",
      success_probability_evidence = "synthesis:sha256:0000000000000000000000000000000000000000000000000000000000000000",
      synthesis_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
    }
  } {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @evidenced_gadget(%arg0) {
      profile = @evidenced_profile
    } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    fabric.protocol_return %next, %ok : !fabric.patch<@c>, i1
  }
  qlx.action @mismatched_gadget_logical : (!qlx.logical_qubit) -> !qlx.logical_qubit {kind = "idle"}
  fabric.objective @mismatched_gadget_objective implements @mismatched_gadget_logical : (!qlx.logical_qubit) -> !qlx.logical_qubit
  fabric.gadget_spec @mismatched_gadget_spec for @mismatched_gadget_objective : () -> () {encodings = []}
  // expected-error @+1 {{gadget signature must exactly match its GadgetSpec realization signature}}
  fabric.gadget @mismatched_gadget(%arg0: !fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
    %ok = arith.constant true
    fabric.return %arg0, %ok : !fabric.patch<@c>, i1
  } {spec = @mismatched_gadget_spec}
  fabric.protocol @bad_probability : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @attempt(%arg0) : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{success_probability must be finite and lie in (0, 1]}}
    %0 = fabric.retry %ok carries (%next) {
      attempt = @attempt,
      max_attempts = 3 : i64,
      success_probability = 0.0 : f64
    } : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @missing_attempt : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @attempt(%arg0) : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{requires an explicit attempt gadget}}
    %0 = fabric.retry %ok carries (%next) {max_attempts = 3 : i64}
      : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @unresolved_attempt : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @unresolved(%arg0) : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{attempt must resolve to fabric.gadget or a contracted fabric.protocol}}
    %0 = fabric.retry %ok carries (%next) {attempt = @unresolved, max_attempts = 3 : i64}
      : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @mismatched_attempt : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @attempt(%arg0) : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{protocol retry attempts are unsupported until protocols carry a typed selection and predicate-provenance contract}}
    %0 = fabric.retry %ok carries (%next) {attempt = @claimed, max_attempts = 3 : i64}
      : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @mismatched_carry : (!fabric.patch<@c>, !fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>, %other: !fabric.patch<@c>):
    %next, %ok = fabric.call @attempt(%arg0) : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{protocol retry attempts are unsupported until protocols carry a typed selection and predicate-provenance contract}}
    %0 = fabric.retry %ok carries (%other) {attempt = @attempt, max_attempts = 3 : i64}
      : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @invalid_exhaustion : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @attempt(%arg0) : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{exhaustion must be report_failure, abort, or return_last}}
    %0 = fabric.retry %ok carries (%next) {attempt = @attempt, exhaustion = "fail", max_attempts = 3 : i64}
      : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @mismatched_probability_evidence : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %next, %ok = fabric.call @evidenced_attempt(%arg0) {
      profile = @evidenced_profile
    } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    %accepted = fabric.all_false %ok : (i1) -> i1
    // expected-error @+1 {{success_probability must equal the value established by its source}}
    %0 = fabric.retry %accepted carries (%next) {
      attempt = @evidenced_attempt,
      max_attempts = 3 : i64,
      profile = @evidenced_profile,
      success_probability = 0.75 : f64,
      success_probability_source = @evidenced_attempt,
      success_probability_evidence = "synthesis:sha256:0000000000000000000000000000000000000000000000000000000000000000"
    } : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
}
