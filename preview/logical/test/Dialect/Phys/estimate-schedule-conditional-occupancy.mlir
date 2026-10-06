// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=independent result=s' \
// RUN:   --phys-estimate-schedule='schedule=s lower-tier=analytical result=e' | \
// RUN:   FileCheck %s --check-prefix=ESTIMATE
// RUN: qlx-opt %s --mlir-disable-threading \
// RUN:   --phys-schedule='graph=independent result=s' \
// RUN:   --phys-estimate-schedule='schedule=s lower-tier=analytical result=e' | \
// RUN:   FileCheck %s --check-prefix=ESTIMATE
// RUN: env QLX_PROFILE_SCHEDULE_ESTIMATE=1 qlx-opt %s \
// RUN:   --mlir-disable-threading \
// RUN:   --phys-schedule='graph=independent result=s' \
// RUN:   --phys-estimate-schedule='schedule=s lower-tier=analytical result=e' 2>&1 | \
// RUN:   FileCheck %s --check-prefix=WORK

// Twelve reused invocations carry twelve independent condition identities and
// disjoint qubit universes. Exact peak occupancy is the sum of their local
// maxima; it must not enumerate 2^12 branch combinations.
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
  fabric.gadget @work() { fabric.return }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 12 : i64, kind = "qubit", native_actions = []
    }
    phys.qec_binding @estimate_binding {
      qec_region = @estimate_qec::@compute, resources = [@qubits]
    }
  }
  qlx.logical_to_qec @estimate_logical_to_qec {
    logical = @estimate_logical, qec = @estimate_qec,
    entries = [{logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @estimate_qec_to_physical {
    qec = @estimate_qec, physical = @arch,
    entries = [{qec = "compute", binding = "estimate_binding",
                resources = ["qubits"]}]
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
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q2 {index = 2 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q3 {index = 3 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q4 {index = 4 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q5 {index = 5 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q6 {index = 6 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q7 {index = 7 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q8 {index = 8 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q9 {index = 9 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q10 {index = 10 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q11 {index = 11 : i64, kind = "qubit", resource_class = @qubits}

  phys.graph @independent on @arch : () ->
      (!phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>,
       !phys.state<@q3>, !phys.state<@q4>, !phys.state<@q5>,
       !phys.state<@q6>, !phys.state<@q7>, !phys.state<@q8>,
       !phys.state<@q9>, !phys.state<@q10>, !phys.state<@q11>) attributes {
        source_protocol = @estimate_source
      } {
    %states:12 = phys.acquire [@q0, @q1, @q2, @q3, @q4, @q5,
                               @q6, @q7, @q8, @q9, @q10, @q11]
      {event_id = "acquire"}
      : !phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>,
        !phys.state<@q3>, !phys.state<@q4>, !phys.state<@q5>,
        !phys.state<@q6>, !phys.state<@q7>, !phys.state<@q8>,
        !phys.state<@q9>, !phys.state<@q10>, !phys.state<@q11>
    %out0 = "phys.call"(%states#0) <{
      callee = @work, event_id = "call0", instance = "root.work.0"
    }> ({
    ^bb0(%state: !phys.state<@q0>):
      %condition = "arith.constant"() {
        event_id = "condition", value = true
      } : () -> i1
      %selected = "cflow.if"(%condition) <{event_id = "if"}> ({
        %then = phys.delay %state {
          duration_ns = 1.0 : f64, event_id = "then"
        } : (!phys.state<@q0>) -> !phys.state<@q0>
        cflow.yield %then : !phys.state<@q0>
      }, {
        %else = phys.delay %state {
          duration_ns = 1.0 : f64, event_id = "else"
        } : (!phys.state<@q0>) -> !phys.state<@q0>
        cflow.yield %else : !phys.state<@q0>
      }) : (i1) -> !phys.state<@q0>
      phys.yield %selected : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    %out1 = phys.call_template %states#1 {
      callee = @work, event_id = "call1", instance = "root.work.1",
      state_aliases = [{alias = @q1, template = @q0}], template_event = "call0"
    } : (!phys.state<@q1>) -> !phys.state<@q1>
    %out2 = phys.call_template %states#2 {
      callee = @work, event_id = "call2", instance = "root.work.2",
      state_aliases = [{alias = @q2, template = @q0}], template_event = "call0"
    } : (!phys.state<@q2>) -> !phys.state<@q2>
    %out3 = phys.call_template %states#3 {
      callee = @work, event_id = "call3", instance = "root.work.3",
      state_aliases = [{alias = @q3, template = @q0}], template_event = "call0"
    } : (!phys.state<@q3>) -> !phys.state<@q3>
    %out4 = phys.call_template %states#4 {
      callee = @work, event_id = "call4", instance = "root.work.4",
      state_aliases = [{alias = @q4, template = @q0}], template_event = "call0"
    } : (!phys.state<@q4>) -> !phys.state<@q4>
    %out5 = phys.call_template %states#5 {
      callee = @work, event_id = "call5", instance = "root.work.5",
      state_aliases = [{alias = @q5, template = @q0}], template_event = "call0"
    } : (!phys.state<@q5>) -> !phys.state<@q5>
    %out6 = phys.call_template %states#6 {
      callee = @work, event_id = "call6", instance = "root.work.6",
      state_aliases = [{alias = @q6, template = @q0}], template_event = "call0"
    } : (!phys.state<@q6>) -> !phys.state<@q6>
    %out7 = phys.call_template %states#7 {
      callee = @work, event_id = "call7", instance = "root.work.7",
      state_aliases = [{alias = @q7, template = @q0}], template_event = "call0"
    } : (!phys.state<@q7>) -> !phys.state<@q7>
    %out8 = phys.call_template %states#8 {
      callee = @work, event_id = "call8", instance = "root.work.8",
      state_aliases = [{alias = @q8, template = @q0}], template_event = "call0"
    } : (!phys.state<@q8>) -> !phys.state<@q8>
    %out9 = phys.call_template %states#9 {
      callee = @work, event_id = "call9", instance = "root.work.9",
      state_aliases = [{alias = @q9, template = @q0}], template_event = "call0"
    } : (!phys.state<@q9>) -> !phys.state<@q9>
    %out10 = phys.call_template %states#10 {
      callee = @work, event_id = "call10", instance = "root.work.10",
      state_aliases = [{alias = @q10, template = @q0}], template_event = "call0"
    } : (!phys.state<@q10>) -> !phys.state<@q10>
    %out11 = phys.call_template %states#11 {
      callee = @work, event_id = "call11", instance = "root.work.11",
      state_aliases = [{alias = @q11, template = @q0}], template_event = "call0"
    } : (!phys.state<@q11>) -> !phys.state<@q11>
    phys.return %out0, %out1, %out2, %out3, %out4, %out5,
                %out6, %out7, %out8, %out9, %out10, %out11
      : !phys.state<@q0>, !phys.state<@q1>, !phys.state<@q2>,
        !phys.state<@q3>, !phys.state<@q4>, !phys.state<@q5>,
        !phys.state<@q6>, !phys.state<@q7>, !phys.state<@q8>,
        !phys.state<@q9>, !phys.state<@q10>, !phys.state<@q11>
  }
}

// ESTIMATE: qlx.estimate_result @e
// ESTIMATE-SAME: peak_active_physical_qubits = 12 : i64
// ESTIMATE-SAME: physical_qubits = 12 : i64
// ESTIMATE-SAME: physical_resources = 12 : i64

// No overlapping-component frontier work is needed for disjoint universes.
// WORK: phys-estimate-schedule peak-shard 0
// WORK-SAME: conditional-work=0
