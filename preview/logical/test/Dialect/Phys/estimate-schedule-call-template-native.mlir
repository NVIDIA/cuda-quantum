// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=shared result=shared_schedule' \
// RUN:   --phys-estimate-schedule='schedule=shared_schedule lower-tier=analytical result=estimate' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=shared result=shared_schedule' \
// RUN:   --phys-estimate-schedule='schedule=shared_schedule lower-tier=analytical result=estimate' \
// RUN:   | FileCheck %s
// RUN: qlx-opt %s --phys-schedule='graph=shared result=shared_schedule' \
// RUN:   | sed 's/binding:factory_binding,qubits\[1\]/qubits[1]/' \
// RUN:   | not qlx-opt 2>&1 | FileCheck %s --check-prefix=MUTATE
// RUN: qlx-opt %s --phys-schedule='graph=global result=global_schedule' \
// RUN:   | FileCheck %s --check-prefix=GLOBAL
// RUN: qlx-opt %s --phys-schedule='graph=global result=global_schedule' \
// RUN:   | sed 's/global.outer1|call_template|10|4|qubits\[0\],qubits\[1\]/global.outer1|call_template|10|4|qubits[1]/' \
// RUN:   | not qlx-opt 2>&1 | FileCheck %s --check-prefix=GLOBAL-MUTATE

module attributes {qlx.profiles = ["p2n", "p3"]} {
  lvm.domain @estimate_logical {
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
                pipeline_depth = "1", physical_qubits = "2"}
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
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
    fabric.region @factory {
      code = @estimate_code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>
    }
  }
  fabric.gadget @estimate_source() { fabric.return }
  fabric.gadget @inner() { fabric.return }
  fabric.gadget @outer() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = []
    }
    phys.resource_class @factory_qubits {
      count = 2 : i64, kind = "qubit", native_actions = []
    }
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@qubits]
    }
    phys.qec_binding @factory_binding {
      qec_region = @estimate_qec::@factory, resources = [@factory_qubits]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = 1.0 : f64}
  }
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"},
               {logical = "factory", qec = "factory"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["qubits"]},
               {qec = "factory", binding = "factory_binding",
                resources = ["factory_qubits"]}]
  }
  qlx.device @estimate_device {
    logical = @estimate_logical, qec = @estimate_qec, physical = @arch,
    logical_to_qec = @estimate_logical_to_qec,
    qec_to_physical = @estimate_qec_to_physical, operating_point = @point
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
  phys.graph @shared on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        operating_point = @point, source_protocol = @estimate_source
      } {
    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %inner0 = "phys.call"(%q0) <{
      callee = @inner, event_id = "inner0", instance = "root.inner.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      %prepared = phys.prepare %state {event_id = "prepare0", state = "zero"}
        : (!phys.state<@q0>) -> (!phys.state<@q0>)
      %request = phys.resource_request "t_state" from @estimate_logical::@magic {
        provider = @provider, region = @estimate_logical::@factory,
        physical_binding = @arch::@factory_binding,
        duration_ns = 1.0 : f64,
        factory_attempt_duration_ns = 1.0 : f64,
        factory_acceptance_probability = 1.0 : f64,
        factory_pipeline_depth = 1 : i64,
        factory_mode = "scheduled_macro", event_id = "request0"
      } : !event.handle<!phys.resource_payload<@t_state>, "linear">
      %payload = event.await %request {event_id = "await0"}
        : !event.handle<!phys.resource_payload<@t_state>, "linear">
          -> !phys.resource_payload<@t_state>
      phys.discard_resource_payload %payload {event_id = "discard0"}
        : !phys.resource_payload<@t_state>
      phys.yield %prepared : !phys.state<@q0>
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
    %outer1 = phys.call_template %q1 {
      callee = @outer, event_id = "outer1", instance = "root.outer.1",
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "outer0"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    phys.return %outer0, %outer1 : !phys.state<@q0>, !phys.state<@q1>
  }

  // A registered opaque binding is unrelated to this reusable call body. The
  // zero-operand barrier synchronizes concrete resources only, and its global
  // timing must cross both compact-call levels without state-alias projection.
  phys.graph @global on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>) attributes {
        operating_point = @point, source_protocol = @estimate_source
      } {
    %request = phys.resource_request "t_state" from @estimate_logical::@magic {
      provider = @provider, region = @estimate_logical::@factory,
      physical_binding = @arch::@factory_binding,
      duration_ns = 1.0 : f64,
      factory_attempt_duration_ns = 1.0 : f64,
      factory_acceptance_probability = 1.0 : f64,
      factory_pipeline_depth = 1 : i64,
      factory_mode = "scheduled_macro", event_id = "global.request"
    } : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %payload = event.await %request {event_id = "global.await"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
        -> !phys.resource_payload<@t_state>
    phys.discard_resource_payload %payload {event_id = "global.discard"}
      : !phys.resource_payload<@t_state>

    %q0, %q1 = phys.acquire [@q0, @q1] {event_id = "global.acquire"}
      : !phys.state<@q0>, !phys.state<@q1>
    %q1_ready = phys.delay %q1 {
      duration_ns = 4.0 : f64, event_id = "global.q1.ready"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %inner0 = "phys.call"(%q0) <{
      callee = @inner, event_id = "global.inner0", instance = "global.inner.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      phys.barrier {domains = ["clock"], event_id = "global.barrier"}
        : () -> ()
      phys.yield %state : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %outer0 = "phys.call"(%inner0) <{
      callee = @outer, event_id = "global.outer0", instance = "global.outer.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      %nested = phys.call_template %state {
        callee = @inner, event_id = "global.inner1",
        instance = "global.outer.inner.0", template_event = "global.inner0"
      } : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %nested : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %q0_held = phys.delay %outer0 {
      duration_ns = 6.0 : f64, event_id = "global.q0.held"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    %outer1 = phys.call_template %q1_ready {
      callee = @outer, event_id = "global.outer1",
      instance = "global.outer.1",
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "global.outer0"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %after = phys.delay %q0_held {
      duration_ns = 1.0 : f64, event_id = "global.after"
    } : (!phys.state<@q0>) -> !phys.state<@q0>
    phys.return %after, %outer1 : !phys.state<@q0>, !phys.state<@q1>
  }
}

// The binding is hidden inside the canonical inner call and is absent from
// both template boundaries. Recursive derivation must carry it through inner1
// while summarizing outer0, then independently authenticate outer1.
// CHECK: phys.schedule @shared_schedule for @shared
// CHECK-SAME: "outer1|call_template|{{[0-9.]+}}|{{[0-9.]+}}|binding:factory_binding,qubits[1]
// MUTATE: 'phys.schedule' op schedule event 'outer1' resources must exactly
// MUTATE-SAME: match its resolved physical resource identities

// Each canonical/template execution accounts for both its state occupancy and
// the hidden two-qubit scheduled-macro binding.
// CHECK: qlx.estimate_result @estimate
// CHECK-SAME: active_physical_qubit_time_ns = 9.000000e+00 : f64
// CHECK-SAME: active_resource_time_ns = 9.000000e+00 : f64
// The outer template executes on q1 while the first canonical inner call uses
// q0.  Reusing q0's canonical qubit set would incorrectly report one.
// CHECK-SAME: peak_active_physical_qubits = 3 : i64
// CHECK-SAME: peak_concurrency = 3 : i64
// CHECK-SAME: physical_qubits = 4 : i64

// The global barrier's canonical first use is four nanoseconds into each
// compact body. global.outer1 therefore waits until 10 so its barrier lands at
// q0's unaliased ready time 14. The unrelated opaque factory binding is absent.
// GLOBAL: phys.schedule @global_schedule for @global
// GLOBAL-SAME: "global.inner1|call_template|4|4|qubits[0],qubits[1]
// GLOBAL-SAME: "global.q0.held|delay|8|6|qubits[0]
// GLOBAL-SAME: "global.outer1|call_template|10|4|qubits[0],qubits[1]|deps=
// GLOBAL-SAME: "global.after|delay|14|1|qubits[0]
// GLOBAL-NOT: global.outer1{{[^\"]*}}binding:factory_binding
// GLOBAL-SAME: makespan_ns = 1.500000e+01 : f64

// GLOBAL-MUTATE: schedule event 'global.outer1' resources must exactly match
// GLOBAL-MUTATE-SAME: its resolved physical resource identities
