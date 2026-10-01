// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=global result=global_schedule' \
// RUN:   | FileCheck %s --check-prefix=POSITIVE
// RUN: qlx-opt %s --phys-schedule='graph=global result=global_schedule' \
// RUN:   | sed 's/outer1|call_template|10|4|qubits\[0\],qubits\[1\]/outer1|call_template|10|4|qubits[1]/' \
// RUN:   | not qlx-opt 2>&1 | FileCheck %s --check-prefix=MUTATION

// A zero-operand clock barrier is global over concrete phys.resource
// identities. Its timing and closure survive two compact-call levels without
// passing through state_aliases. The unrelated factory binding is registered
// as an opaque scheduler resource, but it is not part of the clock domain.

module attributes {qlx.profiles = ["p2n", "p3"]} {
  lvm.domain @logical {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
    lvm.space @factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">],
      capacity = 1 : i64
    }
    lvm.stream @magic {
      backing_region = @factory, capacity = 1 : i64,
      produced_by = @provider,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @t_state
    }
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @produce_t : () -> !fabric.resource<@t_state> {
    kind = "produce_t_state"
  }
  fabric.protocol @provider : () -> !fabric.resource<@t_state> attributes {
    objective = @produce_t,
    metadata = {factory_mode = "scheduled_macro",
                cycles_per_attempt = "1", acceptance_probability = "1.0",
                pipeline_depth = "1", physical_qubits = "1"}
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
  fabric.code @code {
    distance = 1 : i64,
    metadata = {distance_method = "fixture",
                distance_provenance = @code_evidence},
    partitions = {data = 1 : i64}
  }
  fabric.code_profile @code_evidence {
    code = @code, distance_claim = 1 : i64, distance_status = "exact",
    evidence = ["global-barrier-fixture@1"]
  }
  fabric.machine @qec {
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>
    }
    fabric.region @factory {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>
    }
  }
  fabric.gadget @source() { fabric.return }
  fabric.gadget @inner() { fabric.return }
  fabric.gadget @outer() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = []
    }
    phys.resource_class @factory_qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@qubits]
    }
    phys.qec_binding @magic_binding {
      qec_region = @qec::@factory, resources = [@factory_qubits]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = 1.0 : f64}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "compute", qec = "compute"},
               {logical = "factory", qec = "factory"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "compute", binding = "compute_binding",
                resources = ["qubits"]},
               {qec = "factory", binding = "magic_binding",
                resources = ["factory_qubits"]}]
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }

  phys.graph @global on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        operating_point = @point, source_protocol = @source
      } {
    %request = phys.resource_request "t_state" from @logical::@magic {
      provider = @provider, region = @logical::@factory,
      physical_binding = @arch::@magic_binding,
      duration_ns = 1.0 : f64,
      factory_attempt_duration_ns = 1.0 : f64,
      factory_acceptance_probability = 1.0 : f64,
      factory_pipeline_depth = 1 : i64,
      factory_mode = "scheduled_macro", event_id = "request"
    } : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %payload = event.await %request {event_id = "await"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
        -> !phys.resource_payload<@t_state>
    phys.discard_resource_payload %payload {event_id = "discard"}
      : !phys.resource_payload<@t_state>

    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %q1_ready = phys.delay %q1 {
      duration_ns = 4.0 : f64, event_id = "q1.ready"
    } : (!phys.state<@q1>) -> !phys.state<@q1>

    %inner0 = "phys.call"(%q0) <{
      callee = @inner, event_id = "inner0", instance = "root.inner.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      phys.barrier {domains = ["clock"], event_id = "global.barrier"}
        : () -> ()
      phys.yield %state : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>

    %outer0 = "phys.call"(%inner0) <{
      callee = @outer, event_id = "outer0", instance = "root.outer.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      %nested = phys.call_template %state {
        callee = @inner, event_id = "inner1", instance = "outer.inner.0",
        template_event = "inner0"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %nested : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>

    %q0_held = phys.delay %outer0 {
      duration_ns = 6.0 : f64, event_id = "q0.held"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %outer1 = phys.call_template %q1_ready {
      callee = @outer, event_id = "outer1", instance = "root.outer.1",
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "outer0"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %after = phys.delay %q0_held {
      duration_ns = 1.0 : f64, event_id = "after.global"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %after, %outer1 : !phys.state<@q0>, !phys.state<@q1>
  }
}

// inner1 proves first-level propagation. outer1 starts at 10 so its internal
// barrier occurs at 14, after q0.held; the replay then owns q0 until 14 even
// though the ordinary state alias is q0 -> q1. The opaque binding never enters
// either compact barrier envelope.
// POSITIVE: phys.schedule @global_schedule for @global
// POSITIVE-SAME: "inner1|call_template|4|4|qubits[0],qubits[1]
// POSITIVE-SAME: "q0.held|delay|8|6|qubits[0]
// POSITIVE-SAME: "outer1|call_template|10|4|qubits[0],qubits[1]|deps=
// POSITIVE-SAME: "after.global|delay|14|1|qubits[0]
// POSITIVE-SAME: makespan_ns = 1.500000e+01 : f64

// MUTATION: schedule event 'outer1' resources must exactly match its resolved
// MUTATION-SAME: physical resource identities (scheduled=qubits[1];
// MUTATION-SAME: expected=qubits[0], qubits[1])
