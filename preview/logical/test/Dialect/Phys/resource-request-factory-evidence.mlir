// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s --check-prefix=VALID
// RUN: sed 's/factory_mode = "scheduled_macro", event_id/factory_mode = "forged", event_id/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=MODEL
// RUN: sed -e 's/factory_attempt_duration_ns = 10000.0/factory_attempt_duration_ns = 20000.0/' -e 's/duration_ns = 20000.0/duration_ns = 40000.0/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=ATTEMPT
// RUN: sed 's/factory_acceptance_probability = 0.5/factory_acceptance_probability = 0.25/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=ACCEPTANCE
// RUN: sed -e 's/factory_acceptance_probability = 0.5/factory_acceptance_probability = 0.25/' -e 's/duration_ns = 20000.0/duration_ns = 40000.0/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=FORGED-PAIR
// RUN: sed -e 's/factory_pipeline_depth = 1/factory_pipeline_depth = 2/' -e 's/duration_ns = 20000.0/duration_ns = 10000.0/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=DEPTH
// RUN: sed 's/duration_ns = 20000.0/duration_ns = 40000.0/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=DURATION
// RUN: sed 's/physical_qubits = "2"/physical_qubits = "3"/' %s | not qlx-opt 2>&1 | FileCheck %s --check-prefix=FOOTPRINT
// RUN: qlx-opt %s --phys-schedule='graph=graph result=s' \
// RUN:   --phys-estimate-schedule='schedule=s lower-tier=analytical result=e' | \
// RUN:   FileCheck %s --check-prefix=ESTIMATE

module attributes {qlx.profiles = ["p3"]} {
  lvm.domain @logical {
    lvm.space @factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">],
      capacity = 1 : i64
    }
    lvm.stream @magic {
      backing_region = @factory,
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
    phys.resource_class @factory_controller {
      kind = "controller", count = 3 : i64, native_actions = []
    }
    phys.qec_binding @factory_binding {
      qec_region = @qec::@factory,
      resources = [@factory_qubits, @factory_controller]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = 1000.0 : f64}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "factory", qec = "factory"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "factory", binding = "factory_binding",
                resources = ["factory_qubits", "factory_controller"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical, operating_point = @point
  }
  qlx.estimate_result @static {
    assumptions = [], data = {}, device = @device,
    evidence = [@provider], root = @provider,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-counts/v1", tier = "static"
  }
  qlx.estimate_result @analytical {
    assumptions = [], data = {}, device = @device,
    evidence = [@provider], lower_tier = @static, root = @provider,
    metadata = {producer = "fixture", producer_version = "1"},
    schema = "qlx.fabric-estimate/v1", tier = "analytical"
  }
  phys.graph @graph on @arch : () -> () attributes {
    operating_point = @point, source_protocol = @provider
  } {
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
    phys.return
  }
}

// VALID: phys.resource_request "t_state"
// VALID-SAME: duration_ns = 2.000000e+04 : f64
// VALID-SAME: factory_acceptance_probability = 5.000000e-01 : f64
// VALID-SAME: factory_attempt_duration_ns = 1.000000e+04 : f64
// VALID-SAME: factory_mode = "scheduled_macro"
// VALID-SAME: factory_pipeline_depth = 1 : i64

// MODEL: 'phys.resource_request' op factory_mode must equal the retained provider metadata
// ATTEMPT: 'phys.resource_request' op factory_attempt_duration_ns must equal provider cycles_per_attempt times the resolved graph cycle duration
// ACCEPTANCE: 'phys.resource_request' op factory_acceptance_probability must equal the retained provider metadata
// FORGED-PAIR: 'phys.resource_request' op factory_acceptance_probability must equal the retained provider metadata
// DEPTH: 'phys.resource_request' op factory_pipeline_depth must equal the retained provider metadata
// DURATION: 'phys.resource_request' op duration_ns must equal the provider-authenticated expected factory output slot
// FOOTPRINT: 'phys.resource_request' op provider physical_qubits must equal the exact selected binding footprint; binding has 2 qubits but provider declares 3

// The full binding is provisioned physical capacity, while only its qubit-kind
// class contributes to qubit footprint and qubit-time evidence.
// ESTIMATE: qlx.estimate_result @e
// ESTIMATE-SAME: active_physical_qubit_time_ns = 4.000000e+04 : f64
// ESTIMATE-SAME: active_resource_time_ns = 1.000000e+05 : f64
// ESTIMATE-SAME: peak_active_physical_qubits = 2 : i64
// ESTIMATE-SAME: physical_qubits = 2 : i64
// ESTIMATE-SAME: physical_resources = 5 : i64
