// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=g result=g_schedule' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=g result=g_schedule' \
// RUN:   | FileCheck %s

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @canonical_a {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @owner_a {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @owner_b {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @later_a {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @g on @arch : () -> () {
    %canonical_flag = "phys.call"() <{
      callee = @work, event_id = "canonical", instance = "root.work.0"
    }> ({
      %state = phys.acquire [@canonical_a] {event_id = "canonical.acquire"}
        : !phys.state<@canonical_a>
      %done = phys.delay %state {
        duration_ns = 1.0 : f64, event_id = "canonical.delay"
      } : (!phys.state<@canonical_a>) -> !phys.state<@canonical_a>
      phys.release %done {event_id = "canonical.release"}
        : !phys.state<@canonical_a>
      %true = arith.constant true
      phys.yield %true : i1
    }) : () -> i1

    %flag = "phys.call"() <{
      callee = @work, event_id = "owner", instance = "root.work.1"
    }> ({
      %a = phys.acquire [@owner_a] {event_id = "owner.a.acquire"}
        : !phys.state<@owner_a>
      %b = phys.acquire [@owner_b] {event_id = "owner.b.acquire"}
        : !phys.state<@owner_b>
      %a_done = phys.delay %a {
        duration_ns = 9.0 : f64, event_id = "owner.a.delay"
      } : (!phys.state<@owner_a>) -> !phys.state<@owner_a>
      %b_done = phys.delay %b {
        duration_ns = 19.0 : f64, event_id = "owner.b.delay"
      } : (!phys.state<@owner_b>) -> !phys.state<@owner_b>
      phys.release %a_done {event_id = "owner.a.release"}
        : !phys.state<@owner_a>
      phys.release %b_done {event_id = "owner.b.release"}
        : !phys.state<@owner_b>
      %true = arith.constant true
      phys.yield %true : i1
    }) : () -> i1

    // The owner exports qubits[0] at 10 ns while an unrelated qubits[1] tail
    // keeps its envelope alive until 19 ns.  Compact internal reuse is a
    // resource dependency and may begin at 10 ns.
    %template_flag = phys.call_template {
      callee = @work, event_id = "template", instance = "root.work.2",
      template_event = "canonical"
    } : () -> i1

    // This acquire observes both the latest compact effect owner and the
    // explicit allocation-lifetime edge retained below.
    %later = phys.acquire [@later_a] {event_id = "later.acquire"}
      : !phys.state<@later_a>
    %later_done = phys.delay %later {
      duration_ns = 1.0 : f64, event_id = "later.delay"
    } : (!phys.state<@later_a>) -> !phys.state<@later_a>
    phys.release %later_done {event_id = "later.release"}
      : !phys.state<@later_a>

    // A real SSA result from the same owner remains a data dependency and
    // therefore waits for the complete 19 ns call envelope.
    "phys.call"(%flag) <{
      callee = @work, event_id = "consumer", instance = "root.work.3"
    }> ({
    ^bb0(%arg: i1):
      phys.yield
    }) : (i1) -> ()
    phys.return
  }

  phys.allocation_mapping @allocations for @g {entries = [
    {acquire = "canonical.acquire", allocation = "canonical",
     indices = array<i64: 0>, release = "canonical.release",
     resource_class = @qubits, resources = [@canonical_a]},
    {acquire = "owner.a.acquire", after = ["canonical.release"],
     allocation = "owner_a", indices = array<i64: 0>,
     release = "owner.a.release", resource_class = @qubits,
     resources = [@owner_a]},
    {acquire = "owner.b.acquire", allocation = "owner_b",
     indices = array<i64: 1>, release = "owner.b.release",
     resource_class = @qubits, resources = [@owner_b]},
    {acquire = "later.acquire", after = ["owner.a.release"],
     allocation = "later_a", indices = array<i64: 0>,
     release = "later.release", resource_class = @qubits,
     resources = [@later_a]}
  ]}
}

// CHECK: phys.schedule @g_schedule for @g
// CHECK-SAME: "owner|call|0|19|control:owner
// CHECK-SAME: "template|call_template|10|1|qubits[0]
// CHECK-SAME: deps=owner|data_deps=|resource_deps=owner|domain_deps=
// CHECK-SAME: "later.acquire|acquire|11|0|qubits[0]
// CHECK-SAME: deps=template,owner.a.release|data_deps=|resource_deps=template,owner.a.release|domain_deps=
// CHECK-SAME: "consumer|call|19|0|control:consumer
// CHECK-SAME: deps=owner|data_deps=owner|resource_deps=|domain_deps=
// CHECK-SAME: makespan_ns = 1.900000e+01 : f64
