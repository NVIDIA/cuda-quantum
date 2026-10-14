// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s
// RUN: qlx-opt %s --phys-schedule='graph=graph result=s' \
// RUN:   --phys-estimate-schedule='schedule=s lower-tier=analytical result=e' | \
// RUN:   FileCheck %s --check-prefix=ESTIMATE
// RUN: qlx-opt %s --phys-schedule='graph=elided result=elided_s' \
// RUN:   --phys-estimate-schedule='schedule=elided_s lower-tier=analytical result=elided_e' | \
// RUN:   FileCheck %s --check-prefix=ELIDED
// RUN: qlx-opt %s --phys-schedule='graph=elided_alias result=alias_s' \
// RUN:   --phys-estimate-schedule='schedule=alias_s lower-tier=analytical result=alias_e' | \
// RUN:   FileCheck %s --check-prefix=ALIASED

module attributes {qlx.profiles = ["p2n", "p3"]} {
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
  fabric.gadget @work() {
    fabric.return
  }
  phys.machine @arch {
    phys.resource_class @q {count = 2 : i64, kind = "qubit",
      native_actions = []}
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@q]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @q}
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["q"]}]
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
  phys.graph @graph on @arch : () -> () attributes {
    source_protocol = @estimate_source
  } {
    "phys.call"() <{
      callee = @work, event_id = "call0", instance = "root.work.call0"
    }> ({
      phys.yield
    }) : () -> ()
    "phys.call_template"() <{
      callee = @work, event_id = "call1", instance = "root.work.call1",
      template_event = "call0"
    }> : () -> ()
    phys.return
  }
  phys.graph @elided on @arch : () -> () attributes {
    source_protocol = @estimate_source
  } {
    %state = phys.acquire [@q0] {event_id = "elided.acquire"}
      : !phys.state<@q0>
    %next = "phys.call"(%state) <{
      callee = @work, event_id = "elided.canonical",
      instance = "root.work.elided.canonical"
    }> ({
    ^bb0(%arg0: !phys.state<@q0>):
      phys.yield %arg0 : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    "phys.call_template"() <{
      callee = @work, event_id = "elided.template",
      instance = "root.work.elided.template",
      state_boundary_elided, template_event = "elided.canonical"
    }> : () -> ()
    phys.release %next {event_id = "elided.release"}
      : !phys.state<@q0>
    phys.return
  }
  phys.graph @elided_alias on @arch : () -> () attributes {
    source_protocol = @estimate_source
  } {
    %left = phys.acquire [@q0] {event_id = "alias.acquire0"}
      : !phys.state<@q0>
    %right = phys.acquire [@q1] {event_id = "alias.acquire1"}
      : !phys.state<@q1>
    %left_next = "phys.call"(%left) <{
      callee = @work, event_id = "alias.canonical",
      instance = "root.work.alias.canonical"
    }> ({
    ^bb0(%arg0: !phys.state<@q0>):
      phys.yield %arg0 : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    "phys.call_template"() <{
      callee = @work, event_id = "alias.template",
      instance = "root.work.alias.template", state_boundary_elided,
      state_aliases = [{alias = @q1, template = @q0}],
      template_event = "alias.canonical"
    }> : () -> ()
    phys.release %left_next {event_id = "alias.release0"}
      : !phys.state<@q0>
    phys.release %right {event_id = "alias.release1"}
      : !phys.state<@q1>
    phys.return
  }
}

// CHECK: "phys.call"
// CHECK-SAME: event_id = "call0"
// CHECK: phys.call_template
// CHECK-SAME: template_event = "call0"

// Structural control placeholders serialize the empty call envelopes but do
// not become provisioned physical resources.
// ESTIMATE: qlx.estimate_result @e
// ESTIMATE-SAME: physical_qubits = 0 : i64
// ESTIMATE-SAME: physical_resources = 0 : i64

// The compact invocation retains the canonical physical resource effect while
// avoiding a second SSA owner boundary.
// ELIDED: phys.schedule @elided_s
// ELIDED: qlx.estimate_result @elided_e
// ELIDED-SAME: physical_qubits = 2 : i64

// The same compact ABI projects the canonical resource effects through a
// verified state-alias bijection without materializing another SSA boundary.
// ALIASED: phys.schedule @alias_s
// ALIASED: qlx.estimate_result @alias_e
// ALIASED-SAME: physical_qubits = 2 : i64
