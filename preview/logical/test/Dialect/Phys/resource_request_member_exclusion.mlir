// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

// A scheduled macro owns its complete binding footprint. Until P3 has a typed
// aggregate-to-member reservation relation, a graph must not also carry a
// concrete member of any class selected by that binding.
module attributes {qlx.profiles = ["p3"]} {
  lvm.domain @logical {
    lvm.space @factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">],
      capacity = 1 : i64
    }
    lvm.stream @magic {
      backing_region = @factory, produced_by = @provider,
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
                cycles_per_attempt = "10", acceptance_probability = "0.5",
                pipeline_depth = "1", physical_qubits = "2"}
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
  fabric.code @code {
    distance = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @factory {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>
    }
  }
  phys.machine @arch {
    phys.resource_class @factory_qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
    phys.qec_binding @factory_binding {
      qec_region = @qec::@factory, resources = [@factory_qubits]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = 1000.0 : f64}
  }
  phys.resource @factory_q0 {
    kind = "qubit", resource_class = @factory_qubits, index = 0 : i64
  }
  phys.graph @graph on @arch : () -> () attributes {
    operating_point = @point
  } {
    %q0 = phys.acquire [@factory_q0] {event_id = "acquire"}
      : !phys.state<@factory_q0>
    // expected-error @+1 {{scheduled-macro binding resource class @factory_qubits is also used through concrete physical resource @factory_q0; member-level exclusion is unsupported, so scheduled engines must use resource classes disjoint from graph-carried states}}
    %event = phys.resource_request "t_state" from @logical::@magic {
      provider = @provider, region = @factory,
      physical_binding = @arch::@factory_binding,
      duration_ns = 20000.0 : f64,
      factory_attempt_duration_ns = 10000.0 : f64,
      factory_acceptance_probability = 0.5 : f64,
      factory_pipeline_depth = 1 : i64,
      factory_mode = "scheduled_macro", event_id = "request"
    } : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %resource = event.await %event {event_id = "await"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
        -> !phys.resource_payload<@t_state>
    phys.discard_resource_payload %resource {event_id = "discard"}
      : !phys.resource_payload<@t_state>
    phys.release %q0 {event_id = "release"} : !phys.state<@factory_q0>
    phys.return
  }
}
