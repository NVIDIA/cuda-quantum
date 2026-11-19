// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=graph result=scheduled' \
// RUN:   | qlx-opt | FileCheck %s

module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  lvm.domain @logical {
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
    objective = @produce_t
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
    timing = {surface_cycle_ns = "1000"}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "factory", qec = "factory"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{
      qec = "factory", binding = "factory_binding",
      resources = ["factory_qubits"],
      factory_startup_cycles = "20",
      factory_output_interval_cycles = "20",
      factory_model_policy = "guaranteed",
      factory_model_evidence = "user_assertion:test fixture"
    }]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical, operating_point = @point
  }
  phys.factory_model @magic_factory_model {
    buffer_capacity = 1 : i64,
    evidence = "user_assertion:test fixture",
    lane_count = 1 : i64,
    output_interval_ns = 2.000000e+04 : f64,
    operating_point = @point,
    physical_resource_class = @factory_qubits,
    physical_units = 2 : i64,
    policy = "guaranteed",
    provider = @provider,
    provider_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    qec_binding = @factory_binding,
    region = @logical::@factory,
    resource_kind = @t_state,
    startup_ns = 2.000000e+04 : f64,
    stream = @logical::@magic
  }

  phys.graph @graph on @arch : () -> () attributes {
    operating_point = @point, source_protocol = @provider
  } {
    phys.factory_start @magic_factory_model {event_id = "factory_start"}
    "cflow.repeat"() <{count = 2 : i64, event_id = "repeat"}> ({
      %first = phys.resource_request "t_state" from @logical::@magic {
        event_id = "request.first",
        physical_binding = @arch::@factory_binding,
        provider = @provider, region = @logical::@factory,
        factory_model = @magic_factory_model
      } : !event.handle<!phys.resource_payload<@t_state>, "linear">
      %first_payload = event.await %first {event_id = "await.first"}
        : !event.handle<!phys.resource_payload<@t_state>, "linear">
          -> !phys.resource_payload<@t_state>
      phys.discard_resource_payload %first_payload {event_id = "discard.first"}
        : !phys.resource_payload<@t_state>
      phys.barrier {
        domains = ["clock", "factory"], event_id = "factory.barrier"
      } : () -> ()
      %second = phys.resource_request "t_state" from @logical::@magic {
        event_id = "request.second",
        physical_binding = @arch::@factory_binding,
        provider = @provider, region = @logical::@factory,
        factory_model = @magic_factory_model
      } : !event.handle<!phys.resource_payload<@t_state>, "linear">
      %second_payload = event.await %second {event_id = "await.second"}
        : !event.handle<!phys.resource_payload<@t_state>, "linear">
          -> !phys.resource_payload<@t_state>
      phys.discard_resource_payload %second_payload {
        event_id = "discard.second"
      } : !phys.resource_payload<@t_state>
      cflow.yield
    }) : () -> ()
    phys.return
  }
}

// The repeat envelope recursively claims one model and one binding even
// though two requests use them. Re-parsing the generated schedule reruns the
// independent verifier and proves those synthesized names remain stable.
// CHECK: phys.schedule @scheduled for @graph
// CHECK-SAME: "repeat|repeat|{{[0-9.]+}}|{{[0-9.]+}}|factory:magic_factory_model|
// CHECK-SAME: "request.first|resource_request|{{[0-9.]+}}|{{[0-9.]+}}|binding:factory_binding,factory:magic_factory_model
// CHECK-SAME: "factory.barrier|barrier|{{[0-9.]+}}|0|factory:magic_factory_model|deps=request.first|data_deps=|resource_deps=|domain_deps=request.first
// CHECK-SAME: "request.second|resource_request|{{[0-9.]+}}|{{[0-9.]+}}|binding:factory_binding,factory:magic_factory_model
// CHECK-NOT: factory:magic_factory_model,factory:magic_factory_model
