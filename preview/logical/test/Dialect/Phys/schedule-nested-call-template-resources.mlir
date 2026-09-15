// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=nested result=nested_schedule' \
// RUN:   | FileCheck %s --check-prefix=POSITIVE
// RUN: qlx-opt %s --phys-schedule='graph=nested result=nested_schedule' \
// RUN:   | sed 's/outer1|call_template|0|0|qubits\[0\],qubits\[1\]/outer1|call_template|0|0|qubits[0]/' \
// RUN:   | not qlx-opt 2>&1 | FileCheck %s --check-prefix=MUTATION

// A zero-repeat body remains in the schedule as inspectable structure, but it
// contributes no active first-use or availability frontier. Its descendant
// resource union must nevertheless survive through two levels of compact call
// templates. Retaining that union must not turn it into a readiness constraint.

module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget @inner() { fabric.return }
  fabric.gadget @outer() { fabric.return }

  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @scratch_a {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @scratch_z {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @owner_a {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @nested on @arch : () -> () {
    "phys.call"() <{
      callee = @inner, event_id = "inner0", instance = "root.inner.0"
    }> ({
      "cflow.repeat"() <{count = 0 : i64, event_id = "hidden.repeat"}> ({
        %scratch_z, %scratch_a = phys.acquire [@scratch_z, @scratch_a] {
          event_id = "hidden.acquire"
        } : !phys.state<@scratch_z>, !phys.state<@scratch_a>
        %done_z = phys.delay %scratch_z {
          duration_ns = 2.0 : f64, event_id = "hidden.delay_z"
        } : (!phys.state<@scratch_z>) -> !phys.state<@scratch_z>
        %done_a = phys.delay %scratch_a {
          duration_ns = 1.0 : f64, event_id = "hidden.delay_a"
        } : (!phys.state<@scratch_a>) -> !phys.state<@scratch_a>
        phys.release %done_z {event_id = "hidden.release_z"}
          : !phys.state<@scratch_z>
        phys.release %done_a {event_id = "hidden.release_a"}
          : !phys.state<@scratch_a>
        cflow.yield
      }) : () -> ()
      phys.yield
    }) : () -> ()

    "phys.call"() <{
      callee = @outer, event_id = "outer0", instance = "root.outer.0"
    }> ({
      phys.call_template {
        callee = @inner, event_id = "inner1", instance = "outer.inner.0",
        template_event = "inner0"
      } : () -> ()
      phys.yield
    }) : () -> ()

    %owner = phys.acquire [@owner_a] {event_id = "owner.acquire"}
      : !phys.state<@owner_a>
    %owner_done = phys.delay %owner {
      duration_ns = 5.0 : f64, event_id = "owner.delay"
    } : (!phys.state<@owner_a>) -> !phys.state<@owner_a>
    phys.release %owner_done {event_id = "owner.release"}
      : !phys.state<@owner_a>

    phys.call_template {
      callee = @outer, event_id = "outer1", instance = "root.outer.1",
      template_event = "outer0"
    } : () -> ()
    phys.return
  }
}

// Both compact levels retain the canonical descendant union in stable physical
// label order. outer1 still starts at zero while scratch_a is owned until five:
// the union is proof evidence, not an active first-use readiness constraint.
// POSITIVE: phys.schedule @nested_schedule for @nested
// POSITIVE-SAME: "inner1|call_template|0|0|qubits[0],qubits[1]
// POSITIVE-SAME: "owner.delay|delay|0|5|qubits[0]
// POSITIVE-SAME: "outer1|call_template|0|0|qubits[0],qubits[1]|deps=|data_deps=|resource_deps=|domain_deps=
// POSITIVE-SAME: makespan_ns = 5.000000e+00 : f64

// Removing one nested hidden resource from the compact envelope must fail the
// independent recursive provenance proof.
// MUTATION: schedule event 'outer1' resources must exactly match its resolved physical resource identities (scheduled=qubits[0]; expected=qubits[0], qubits[1])
