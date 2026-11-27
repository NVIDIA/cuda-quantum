// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=compact result=compact_schedule' \
// RUN:   | FileCheck %s --check-prefix=COMPACT
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=compact result=compact_schedule' \
// RUN:   | FileCheck %s --check-prefix=COMPACT
// RUN: qlx-opt %s --phys-schedule='graph=explicit result=explicit_schedule' \
// RUN:   | FileCheck %s --check-prefix=EXPLICIT
// RUN: env QLX_PROFILE_P2_TO_P3=1 qlx-opt %s \
// RUN:   --phys-schedule='graph=compact result=compact_schedule' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WORK

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 3 : i64, kind = "qubit", native_actions = []
    }
  }

  // Distinct allocation identities deliberately reuse the same concrete
  // class/index at disjoint times.  The scheduler must exclude by the concrete
  // binding rather than by these symbols.
  phys.resource @compact_prefix {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @compact_scratch0 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @compact_gap {
    index = 2 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @compact_scratch1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @compact_owner {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @compact on @arch : () -> () {
    "phys.call"() <{
      callee = @work, event_id = "canonical", instance = "root.work.0"
    }> ({
      %prefix = phys.acquire [@compact_prefix] {event_id = "prefix.acquire"}
        : !phys.state<@compact_prefix>
      %prefix_done = phys.delay %prefix {
        duration_ns = 3.0 : f64, event_id = "prefix.delay"
      } : (!phys.state<@compact_prefix>) -> !phys.state<@compact_prefix>
      phys.barrier {domains = ["clock"], event_id = "tick0"} : () -> ()

      %scratch0 = phys.acquire [@compact_scratch0] {
        event_id = "scratch0.acquire"
      } : !phys.state<@compact_scratch0>
      %scratch0_done = phys.delay %scratch0 {
        duration_ns = 2.0 : f64, event_id = "scratch0.delay"
      } : (!phys.state<@compact_scratch0>) ->
          !phys.state<@compact_scratch0>
      phys.release %scratch0_done {event_id = "scratch0.release"}
        : !phys.state<@compact_scratch0>
      phys.barrier {domains = ["clock"], event_id = "tick1"} : () -> ()

      %gap = phys.acquire [@compact_gap] {event_id = "gap.acquire"}
        : !phys.state<@compact_gap>
      %gap_done = phys.delay %gap {
        duration_ns = 2.0 : f64, event_id = "gap.delay"
      } : (!phys.state<@compact_gap>) -> !phys.state<@compact_gap>
      phys.barrier {domains = ["clock"], event_id = "tick2"} : () -> ()

      %scratch1 = phys.acquire [@compact_scratch1] {
        event_id = "scratch1.acquire"
      } : !phys.state<@compact_scratch1>
      %scratch1_done = phys.delay %scratch1 {
        duration_ns = 1.0 : f64, event_id = "scratch1.delay"
      } : (!phys.state<@compact_scratch1>) ->
          !phys.state<@compact_scratch1>
      phys.release %scratch1_done {event_id = "scratch1.release"}
        : !phys.state<@compact_scratch1>
      %prefix_final = phys.delay %prefix_done {
        duration_ns = 1.0 : f64, event_id = "prefix.final"
      } : (!phys.state<@compact_prefix>) -> !phys.state<@compact_prefix>
      phys.release %prefix_final {event_id = "prefix.release"}
        : !phys.state<@compact_prefix>
      %gap_final = phys.delay %gap_done {
        duration_ns = 1.0 : f64, event_id = "gap.final"
      } : (!phys.state<@compact_gap>) -> !phys.state<@compact_gap>
      phys.release %gap_final {event_id = "gap.release"}
        : !phys.state<@compact_gap>
      phys.barrier {domains = ["clock"], event_id = "tick3"} : () -> ()
      phys.yield
    }) : () -> ()

    %owner = phys.acquire [@compact_owner] {event_id = "owner.acquire"}
      : !phys.state<@compact_owner>
    %owner_done = phys.delay %owner {
      duration_ns = 5.0 : f64, event_id = "owner.delay"
    } : (!phys.state<@compact_owner>) -> !phys.state<@compact_owner>
    phys.release %owner_done {event_id = "owner.release"}
      : !phys.state<@compact_owner>

    phys.call_template  {
      callee = @work, event_id = "template", instance = "root.work.1",
      template_event = "canonical"
    } : () -> ()
    phys.return
  }

  // The explicit reference uses distinct allocation symbols with the same
  // concrete indices.  Its body is intentionally identical to the canonical
  // body above, including a delayed first scratch use and a later scratch
  // transition separated by a real gap.
  phys.resource @explicit_prefix0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_scratch0 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_gap0 {
    index = 2 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_scratch1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_owner {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_prefix1 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_scratch2 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_gap1 {
    index = 2 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @explicit_scratch3 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @explicit on @arch : () -> () {
    "phys.call"() <{
      callee = @work, event_id = "explicit.canonical",
      instance = "explicit.work.0"
    }> ({
      %prefix = phys.acquire [@explicit_prefix0] {
        event_id = "explicit.prefix0.acquire"
      } : !phys.state<@explicit_prefix0>
      %prefix_done = phys.delay %prefix {
        duration_ns = 3.0 : f64, event_id = "explicit.prefix0.delay"
      } : (!phys.state<@explicit_prefix0>) ->
          !phys.state<@explicit_prefix0>
      phys.barrier {domains = ["clock"], event_id = "explicit.tick0"}
        : () -> ()
      %scratch0 = phys.acquire [@explicit_scratch0] {
        event_id = "explicit.scratch0.acquire"
      } : !phys.state<@explicit_scratch0>
      %scratch0_done = phys.delay %scratch0 {
        duration_ns = 2.0 : f64, event_id = "explicit.scratch0.delay"
      } : (!phys.state<@explicit_scratch0>) ->
          !phys.state<@explicit_scratch0>
      phys.release %scratch0_done {event_id = "explicit.scratch0.release"}
        : !phys.state<@explicit_scratch0>
      phys.barrier {domains = ["clock"], event_id = "explicit.tick1"}
        : () -> ()
      %gap = phys.acquire [@explicit_gap0] {
        event_id = "explicit.gap0.acquire"
      } : !phys.state<@explicit_gap0>
      %gap_done = phys.delay %gap {
        duration_ns = 2.0 : f64, event_id = "explicit.gap0.delay"
      } : (!phys.state<@explicit_gap0>) -> !phys.state<@explicit_gap0>
      phys.barrier {domains = ["clock"], event_id = "explicit.tick2"}
        : () -> ()
      %scratch1 = phys.acquire [@explicit_scratch1] {
        event_id = "explicit.scratch1.acquire"
      } : !phys.state<@explicit_scratch1>
      %scratch1_done = phys.delay %scratch1 {
        duration_ns = 1.0 : f64, event_id = "explicit.scratch1.delay"
      } : (!phys.state<@explicit_scratch1>) ->
          !phys.state<@explicit_scratch1>
      phys.release %scratch1_done {event_id = "explicit.scratch1.release"}
        : !phys.state<@explicit_scratch1>
      %prefix_final = phys.delay %prefix_done {
        duration_ns = 1.0 : f64, event_id = "explicit.prefix0.final"
      } : (!phys.state<@explicit_prefix0>) ->
          !phys.state<@explicit_prefix0>
      phys.release %prefix_final {event_id = "explicit.prefix0.release"}
        : !phys.state<@explicit_prefix0>
      %gap_final = phys.delay %gap_done {
        duration_ns = 1.0 : f64, event_id = "explicit.gap0.final"
      } : (!phys.state<@explicit_gap0>) -> !phys.state<@explicit_gap0>
      phys.release %gap_final {event_id = "explicit.gap0.release"}
        : !phys.state<@explicit_gap0>
      phys.barrier {domains = ["clock"], event_id = "explicit.second.tick0"}
        : () -> ()
      phys.yield
    }) : () -> ()

    %owner = phys.acquire [@explicit_owner] {
      event_id = "explicit.owner.acquire"
    } : !phys.state<@explicit_owner>
    %owner_done = phys.delay %owner {
      duration_ns = 5.0 : f64, event_id = "explicit.owner.delay"
    } : (!phys.state<@explicit_owner>) -> !phys.state<@explicit_owner>
    phys.release %owner_done {event_id = "explicit.owner.release"}
      : !phys.state<@explicit_owner>

    "phys.call"() <{
      callee = @work, event_id = "explicit.second",
      instance = "explicit.work.1"
    }> ({
      %prefix = phys.acquire [@explicit_prefix1] {
        event_id = "explicit.prefix1.acquire"
      } : !phys.state<@explicit_prefix1>
      %prefix_done = phys.delay %prefix {
        duration_ns = 3.0 : f64, event_id = "explicit.prefix1.delay"
      } : (!phys.state<@explicit_prefix1>) ->
          !phys.state<@explicit_prefix1>
      phys.barrier {domains = ["clock"], event_id = "explicit.tick3"}
        : () -> ()
      %scratch2 = phys.acquire [@explicit_scratch2] {
        event_id = "explicit.scratch2.acquire"
      } : !phys.state<@explicit_scratch2>
      %scratch2_done = phys.delay %scratch2 {
        duration_ns = 2.0 : f64, event_id = "explicit.scratch2.delay"
      } : (!phys.state<@explicit_scratch2>) ->
          !phys.state<@explicit_scratch2>
      phys.release %scratch2_done {event_id = "explicit.scratch2.release"}
        : !phys.state<@explicit_scratch2>
      phys.barrier {domains = ["clock"], event_id = "explicit.second.tick1"}
        : () -> ()
      %gap = phys.acquire [@explicit_gap1] {
        event_id = "explicit.gap1.acquire"
      } : !phys.state<@explicit_gap1>
      %gap_done = phys.delay %gap {
        duration_ns = 2.0 : f64, event_id = "explicit.gap1.delay"
      } : (!phys.state<@explicit_gap1>) -> !phys.state<@explicit_gap1>
      phys.barrier {domains = ["clock"], event_id = "explicit.second.tick2"}
        : () -> ()
      %scratch3 = phys.acquire [@explicit_scratch3] {
        event_id = "explicit.scratch3.acquire"
      } : !phys.state<@explicit_scratch3>
      %scratch3_done = phys.delay %scratch3 {
        duration_ns = 1.0 : f64, event_id = "explicit.scratch3.delay"
      } : (!phys.state<@explicit_scratch3>) ->
          !phys.state<@explicit_scratch3>
      phys.release %scratch3_done {event_id = "explicit.scratch3.release"}
        : !phys.state<@explicit_scratch3>
      %prefix_final = phys.delay %prefix_done {
        duration_ns = 1.0 : f64, event_id = "explicit.prefix1.final"
      } : (!phys.state<@explicit_prefix1>) ->
          !phys.state<@explicit_prefix1>
      phys.release %prefix_final {event_id = "explicit.prefix1.release"}
        : !phys.state<@explicit_prefix1>
      %gap_final = phys.delay %gap_done {
        duration_ns = 1.0 : f64, event_id = "explicit.gap1.final"
      } : (!phys.state<@explicit_gap1>) -> !phys.state<@explicit_gap1>
      phys.release %gap_final {event_id = "explicit.gap1.release"}
        : !phys.state<@explicit_gap1>
      phys.barrier {domains = ["clock"], event_id = "explicit.second.tick3"}
        : () -> ()
      phys.yield
    }) : () -> ()
    phys.return
  }
}

// The prior owner remains active from 8 to 13 ns.  The compact envelope may
// start at 10 ns because the canonical body's first use of qubits[1] is three
// nanoseconds into the template.  Its final effect is translated to 18 ns.
// COMPACT: phys.schedule @compact_schedule for @compact
// COMPACT-SAME: "owner.delay|delay|8|5|qubits[1]
// COMPACT-SAME: "template|call_template|10|8|qubits[0],qubits[1],qubits[2]
// COMPACT-SAME: data_deps=|resource_deps=canonical|domain_deps=canonical
// COMPACT-SAME: makespan_ns = 1.800000e+01 : f64

// Explicit expansion retains different envelope timing but the same exact
// physical finish/makespan: the prefix overlaps the prior scratch owner, then
// the first scratch transition begins at 13 ns and the second ends at 18 ns.
// EXPLICIT: phys.schedule @explicit_schedule for @explicit
// EXPLICIT-SAME: "explicit.second|call|8|10|control:explicit.second
// EXPLICIT-SAME: "explicit.scratch2.delay|delay|13|2|qubits[1]
// EXPLICIT-SAME: "explicit.scratch3.delay|delay|17|1|qubits[1]
// EXPLICIT-SAME: makespan_ns = 1.800000e+01 : f64

// The canonical body contains 18 descendants over three physical identities,
// but the rollback proof records only the seven real resource transitions;
// row-only control placeholders never enter the physical frontier.
// WORK: phys-schedule: verifier-frontier state-copies=0 journal-touches=7
