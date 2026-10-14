// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.gadget_spec @retry_spec for @objective
      : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1) {
    encodings = [],
    outcome_map = {
      records = ["ready.outcome"], rows = dense<1> : tensor<1x1xi1>,
      constants = array<i64: 0>, input_syndromes = [[]],
      roles = [["success"]]
    },
    record_schema = ["ready.outcome"]
  }
  fabric.gadget @retry_attempt(%patch: !fabric.patch<@c>)
      -> (!fabric.patch<@c>, i1) {
    %next, %ready = fabric.measure_product %patch {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "ready"
    } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    fabric.return %next, %ready : !fabric.patch<@c>, i1
  } {realization_boundary = {}, spec = @retry_spec}
  fabric.gadget_profile @retry_profile for @retry_attempt {
    fabric.success {records = ["retry_attempt.ready.outcome"]}
  }
  fabric.protocol @double_h : (!fabric.patch<@c>) -> !fabric.patch<@c> attributes {objective = #qlx.action<idle>} {
  ^bb0(%arg0: !fabric.patch<@c>):
    %0 = fabric.call @h(%arg0) : (!fabric.patch<@c>) -> !fabric.patch<@c>
    %1 = fabric.call @h(%0) : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %1 : !fabric.patch<@c>
  }
  fabric.protocol @retrying : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %attempt, %ok = fabric.call @retry_attempt(%arg0) {
      profile = @retry_profile
    } : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    %accepted = fabric.all_false %ok : (i1) -> i1
    %0 = fabric.retry %accepted carries (%attempt) {
      attempt = @retry_attempt, max_attempts = 3 : i64,
      profile = @retry_profile
    }
      : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }
  fabric.protocol @committed : () -> () attributes {
    metadata = {
      input_p1 = "placed_kernel",
      qec_selection_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  } {
    fabric.protocol_return
  }
}

// CHECK: fabric.protocol @double_h : (!fabric.patch<@c>) -> !fabric.patch<@c>
// CHECK-SAME: objective = #qlx.action<idle>
// CHECK: %[[A:.+]] = fabric.call @h(%arg0)
// CHECK: %[[B:.+]] = fabric.call @h(%[[A]])
// CHECK: fabric.protocol_return %[[B]]
// CHECK: fabric.protocol @retrying
// CHECK: fabric.retry
// CHECK-SAME: max_attempts = 3 : i64
// CHECK: fabric.protocol @committed
// CHECK-SAME: input_p1 = "placed_kernel"
// CHECK-SAME: qec_selection_sha256 =
