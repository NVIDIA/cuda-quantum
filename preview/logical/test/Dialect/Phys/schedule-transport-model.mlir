// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --phys-schedule='graph=serial result=serial_schedule' \
// RUN:   | qlx-opt | FileCheck %s --check-prefix=SERIAL
// RUN: qlx-opt %s --phys-schedule='graph=pipelined result=pipelined_schedule' \
// RUN:   | qlx-opt | FileCheck %s --check-prefix=PIPELINED
// RUN: qlx-opt %s --phys-schedule='graph=composed result=composed_schedule' \
// RUN:   | qlx-opt | FileCheck %s --check-prefix=COMPOSED

module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [], capacity = 1 : i64}
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
    lvm.stream @magic_stream {
      backing_region = @factory, capacity = 1 : i64,
      produced_by = @producer,
      produced_by_sha256 = "sha256:7777777777777777777777777777777777777777777777777777777777777777",
      produces = @magic
    }
    lvm.channel @supply {
      from = @factory, to = @magic_stream,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
      direction = "forward", concurrency = 2 : i64
    }
    lvm.channel @delivery {
      from = @magic_stream, to = @compute,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
      direction = "forward", concurrency = 2 : i64
    }
  }
  qlx.action @produce_magic : () -> !fabric.resource<@magic> {
    kind = "produce_magic"
  }
  fabric.protocol @producer : () -> !fabric.resource<@magic> attributes {
    objective = @produce_magic
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @magic
    } : !fabric.resource<@magic>
    fabric.protocol_return %state : !fabric.resource<@magic>
  }
  fabric.protocol @delivery_protocol :
      (!fabric.resource<@magic>) -> !fabric.resource<@magic> attributes {
    component_source_sha256 = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
  } {
  ^bb0(%state: !fabric.resource<@magic>):
    fabric.protocol_return %state : !fabric.resource<@magic>
  }
  fabric.code @code {
    distance = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @factory {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>, block_capacity = 1 : i64
    }
    fabric.region @compute {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<compute>, block_capacity = 1 : i64
    }
    fabric.interconnect @delivery {
      region_a = @factory, port_a = 0 : i64,
      region_b = @compute, port_b = 0 : i64,
      logical_channel = @logical::@delivery,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
      direction = "forward", concurrency = 2 : i64,
      provider = "test@1",
      port_a_name = "factory_port", port_b_name = "compute_port",
      port_a_concurrency = 2 : i64, port_b_concurrency = 2 : i64,
      port_a_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
      port_b_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
      port_a_provider = "test.port@1", port_b_provider = "test.port@1",
      protocol = @delivery_protocol
    }
  }
  phys.machine @arch {
    phys.resource_class @factory_qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
    phys.resource_class @links {
      kind = "transport_lane", count = 2 : i64,
      granularity = "carrier", native_actions = []
    }
    phys.qec_channel_binding @delivery_physical {
      qec_channel = @qec::@delivery, resources = [@links]
    }
    phys.qec_binding @factory_binding {
      qec_region = @qec::@factory, resources = [@factory_qubits]
    }
    phys.qec_binding @compute_binding {
      qec_region = @qec::@compute, resources = [@links]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = "1", surface_cycle_ns = "1"}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "factory", qec = "factory"},
               {logical = "compute", qec = "compute"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "factory", binding = "factory_binding",
                resources = ["factory_qubits"],
                factory_startup_cycles = "7",
                factory_output_interval_cycles = "7",
                factory_model_policy = "guaranteed",
                factory_model_evidence = "user_assertion:test composed factory"},
               {qec = "compute", binding = "compute_binding",
                resources = ["links"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical, operating_point = @point
  }

  phys.transport_model @serial_model {
    architecture = @arch, operating_point = @point,
    qec_binding = @arch::@delivery_physical,
    qec_channel = @qec::@delivery, protocol = @delivery_protocol,
    latency_ns = 5.000000e+00 : f64,
    initiation_interval_ns = 2.000000e+00 : f64,
    interval_semantics = "pipelined", policy = "guaranteed",
    source_endpoint_occupancy = 2 : i64,
    destination_endpoint_occupancy = 2 : i64,
    resource_claims = [{resource_class = @arch::@links, offset = 0 : i64,
                        count = 2 : i64, units = 1 : i64}],
    provider = "qlx.user.transport", provider_version = "1",
    evidence = "user_assertion:test serial transport",
    channel_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    realization_sha256 = "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    protocol_sha256 = "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    binding_sha256 = "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    architecture_sha256 = "sha256:4444444444444444444444444444444444444444444444444444444444444444",
    model_sha256 = "5555555555555555555555555555555555555555555555555555555555555555",
    timing_profile = {cycle_ns = 1.000000e+00 : f64,
                      surface_cycle_ns = 1.000000e+00 : f64}
  }
  phys.factory_model @magic_factory_model {
    buffer_capacity = 1 : i64,
    evidence = "user_assertion:test composed factory",
    lane_count = 1 : i64,
    output_interval_ns = 7.000000e+00 : f64,
    operating_point = @point,
    physical_resource_class = @factory_qubits,
    physical_units = 2 : i64,
    policy = "guaranteed",
    provider = @producer,
    provider_sha256 = "sha256:7777777777777777777777777777777777777777777777777777777777777777",
    qec_binding = @factory_binding,
    region = @logical::@factory,
    resource_kind = @magic,
    startup_ns = 7.000000e+00 : f64,
    stream = @logical::@magic_stream
  }
  phys.transport_model @pipelined_model {
    architecture = @arch, operating_point = @point,
    qec_binding = @arch::@delivery_physical,
    qec_channel = @qec::@delivery, protocol = @delivery_protocol,
    latency_ns = 5.000000e+00 : f64,
    initiation_interval_ns = 2.000000e+00 : f64,
    interval_semantics = "pipelined", policy = "guaranteed",
    source_endpoint_occupancy = 1 : i64,
    destination_endpoint_occupancy = 1 : i64,
    resource_claims = [{resource_class = @arch::@links, offset = 0 : i64,
                        count = 2 : i64, units = 1 : i64}],
    provider = "qlx.user.transport", provider_version = "1",
    evidence = "user_assertion:test pipelined transport",
    channel_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    realization_sha256 = "sha256:1111111111111111111111111111111111111111111111111111111111111111",
    protocol_sha256 = "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    binding_sha256 = "sha256:3333333333333333333333333333333333333333333333333333333333333333",
    architecture_sha256 = "sha256:4444444444444444444444444444444444444444444444444444444444444444",
    model_sha256 = "6666666666666666666666666666666666666666666666666666666666666666",
    timing_profile = {cycle_ns = 1.000000e+00 : f64,
                      surface_cycle_ns = 1.000000e+00 : f64}
  }

  phys.graph @serial on @arch : () -> () attributes {
    operating_point = @point, source_protocol = @delivery_protocol
  } {
    %first = phys.produce_resource @magic at @factory {
      event_id = "produce.first", protocol = @delivery_protocol
    } : !phys.resource_payload<@magic>
    %first_out = phys.transport_resource %first from @factory to @compute {
      event_id = "transport.first", model = @serial_model,
      protocol = @delivery_protocol
    } : !phys.resource_payload<@magic> -> !phys.resource_payload<@magic>
    phys.discard_resource_payload %first_out {event_id = "discard.first"}
      : !phys.resource_payload<@magic>
    %second = phys.produce_resource @magic at @factory {
      event_id = "produce.second", protocol = @delivery_protocol
    } : !phys.resource_payload<@magic>
    %second_out = phys.transport_resource %second from @factory to @compute {
      event_id = "transport.second", model = @serial_model,
      protocol = @delivery_protocol
    } : !phys.resource_payload<@magic> -> !phys.resource_payload<@magic>
    phys.discard_resource_payload %second_out {event_id = "discard.second"}
      : !phys.resource_payload<@magic>
    phys.return
  }

  phys.graph @pipelined on @arch : () -> () attributes {
    operating_point = @point, source_protocol = @delivery_protocol
  } {
    %first = phys.produce_resource @magic at @factory {
      event_id = "produce.first", protocol = @delivery_protocol
    } : !phys.resource_payload<@magic>
    %first_out = phys.transport_resource %first from @factory to @compute {
      event_id = "transport.first", model = @pipelined_model,
      protocol = @delivery_protocol
    } : !phys.resource_payload<@magic> -> !phys.resource_payload<@magic>
    phys.discard_resource_payload %first_out {event_id = "discard.first"}
      : !phys.resource_payload<@magic>
    %second = phys.produce_resource @magic at @factory {
      event_id = "produce.second", protocol = @delivery_protocol
    } : !phys.resource_payload<@magic>
    %second_out = phys.transport_resource %second from @factory to @compute {
      event_id = "transport.second", model = @pipelined_model,
      protocol = @delivery_protocol
    } : !phys.resource_payload<@magic> -> !phys.resource_payload<@magic>
    phys.discard_resource_payload %second_out {event_id = "discard.second"}
      : !phys.resource_payload<@magic>
    phys.return
  }

  phys.graph @composed on @arch : () -> () attributes {
    operating_point = @point, source_protocol = @delivery_protocol
  } {
    phys.factory_start @magic_factory_model {event_id = "factory.start"}
    %first = phys.resource_request "magic" from @logical::@magic_stream {
      event_id = "request.first", factory_model = @magic_factory_model,
      physical_binding = @arch::@factory_binding,
      provider = @producer, region = @logical::@factory
    } : !event.handle<!phys.resource_payload<@magic>, "linear">
    %first_payload = event.await %first {event_id = "await.first"}
      : !event.handle<!phys.resource_payload<@magic>, "linear">
        -> !phys.resource_payload<@magic>
    %first_out = phys.transport_resource %first_payload from @factory to @compute {
      event_id = "transport.first", model = @pipelined_model,
      protocol = @delivery_protocol
    } : !phys.resource_payload<@magic> -> !phys.resource_payload<@magic>
    phys.discard_resource_payload %first_out {event_id = "discard.first"}
      : !phys.resource_payload<@magic>
    %second = phys.resource_request "magic" from @logical::@magic_stream {
      event_id = "request.second", factory_model = @magic_factory_model,
      physical_binding = @arch::@factory_binding,
      provider = @producer, region = @logical::@factory
    } : !event.handle<!phys.resource_payload<@magic>, "linear">
    %second_payload = event.await %second {event_id = "await.second"}
      : !event.handle<!phys.resource_payload<@magic>, "linear">
        -> !phys.resource_payload<@magic>
    %second_out = phys.transport_resource %second_payload from @factory to @compute {
      event_id = "transport.second", model = @pipelined_model,
      protocol = @delivery_protocol
    } : !phys.resource_payload<@magic> -> !phys.resource_payload<@magic>
    phys.discard_resource_payload %second_out {event_id = "discard.second"}
      : !phys.resource_payload<@magic>
    phys.return
  }
}

// One endpoint lane forces the second transfer to wait for full latency.
// SERIAL: phys.schedule @serial_schedule for @serial
// SERIAL-SAME: "transport.first|transport_resource|1|5|{{[^\"]*}}links[0]
// SERIAL-SAME: "transport.second|transport_resource|6|5|{{[^\"]*}}links[1]

// Two endpoint/resource lanes admit the second transfer at the typed II.
// PIPELINED: phys.schedule @pipelined_schedule for @pipelined
// PIPELINED-SAME: "transport.first|transport_resource|1|5|{{[^\"]*}}links[0]
// PIPELINED-SAME: "transport.second|transport_resource|3|5|{{[^\"]*}}links[1]

// Ordinary scheduler composition preserves the factory's seven-cycle output
// cadence and then acquires each link/endpoint slice exactly once.  Factory
// qubits and transport members remain disjoint typed claims in one physical
// architecture rather than duplicated device-footprint declarations.
// COMPOSED: phys.schedule @composed_schedule for @composed
// COMPOSED-SAME: "factory.start|factory_start|0|7|factory:magic_factory_model
// COMPOSED-SAME: "request.first|resource_request|7|0|binding:factory_binding,factory:magic_factory_model
// COMPOSED-SAME: "transport.first|transport_resource|8|5|{{[^\"]*}}links[0]
// COMPOSED-SAME: "request.second|resource_request|14|0|binding:factory_binding,factory:magic_factory_model
// COMPOSED-SAME: "transport.second|transport_resource|15|5|{{[^\"]*}}links[1]
