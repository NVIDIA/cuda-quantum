// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @q {
      kind = "qubit", count = 3 : i64, native_actions = []
    }
  }

  // Resource symbols identify distinct allocation instances. Reusing index 0
  // is legal only when SSA order between the enclosing calls proves that the
  // first nested lifetime has ended before the second one starts.
  phys.resource @first {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @second {kind = "qubit", resource_class = @q, index = 0 : i64}
  phys.resource @anchor {kind = "qubit", resource_class = @q, index = 1 : i64}
  phys.graph @reuse on @arch : () -> () {
    %0 = phys.acquire [@anchor] {event_id = "acquire_anchor"}
      : !phys.state<@anchor>
    %1 = "phys.call"(%0) <{callee = @first_user,
                            instance = "reuse.first",
                            event_id = "call0"}> ({
    ^bb0(%current: !phys.state<@anchor>):
      %scratch = phys.acquire [@first] {event_id = "acquire0"}
        : !phys.state<@first>
      phys.release %scratch {event_id = "release0"} : !phys.state<@first>
      phys.yield %current : !phys.state<@anchor>
    }) : (!phys.state<@anchor>) -> !phys.state<@anchor>
    %2 = "phys.call"(%1) <{callee = @second_user,
                            instance = "reuse.second",
                            event_id = "call1"}> ({
    ^bb0(%current: !phys.state<@anchor>):
      %scratch = phys.acquire [@second] {event_id = "acquire1"}
        : !phys.state<@second>
      phys.release %scratch {event_id = "release1"} : !phys.state<@second>
      phys.yield %current : !phys.state<@anchor>
    }) : (!phys.state<@anchor>) -> !phys.state<@anchor>
    phys.release %2 {event_id = "release_anchor"} : !phys.state<@anchor>
    phys.return
  }
  phys.allocation_mapping @reuse_allocations for @reuse {
    entries = [
      {allocation = "call0.scratch", resource_class = @q,
       resources = [@first], indices = array<i64: 0>,
       acquire = "acquire0", release = "release0"},
      {allocation = "call1.scratch", resource_class = @q,
       resources = [@second], indices = array<i64: 0>,
       acquire = "acquire1", release = "release1"},
      {allocation = "anchor", resource_class = @q,
       resources = [@anchor], indices = array<i64: 1>,
       acquire = "acquire_anchor", release = "release_anchor"}
    ]
  }

  // Allocation membership is a resource set. Physical transformations may
  // permute the states before their common lifetime ends.
  phys.resource @left {kind = "qubit", resource_class = @q, index = 1 : i64}
  phys.resource @right {kind = "qubit", resource_class = @q, index = 2 : i64}
  phys.graph @permuted_release on @arch : () -> () {
    %0:2 = phys.acquire [@left, @right] {event_id = "acquire2"}
      : !phys.state<@left>, !phys.state<@right>
    phys.release %0#1, %0#0 {event_id = "release2"}
      : !phys.state<@right>, !phys.state<@left>
    phys.return
  }
  phys.allocation_mapping @permuted_release_allocations for @permuted_release {
    entries = [{allocation = "permuted", resource_class = @q,
      resources = [@left, @right], indices = array<i64: 1, 2>,
      acquire = "acquire2", release = "release2"}]
  }
}

// CHECK: phys.resource @first
// CHECK: phys.resource @second
// CHECK: phys.resource @anchor
// CHECK: phys.acquire [@first] {event_id = "acquire0"}
// CHECK: phys.release
// CHECK: phys.acquire [@second] {event_id = "acquire1"}
// CHECK: phys.allocation_mapping @reuse_allocations for @reuse
// CHECK: allocation = "call0.scratch"
// CHECK: allocation = "call1.scratch"
// CHECK: phys.allocation_mapping @permuted_release_allocations for @permuted_release
