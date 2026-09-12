// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=exact device-symbol=device graph-symbol=exact_physical})' | FileCheck %s --check-prefix=EXACT
// RUN: not qlx-opt %s --pass-pipeline='builtin.module(fabric-to-phys{root-symbol=pooled device-symbol=device graph-symbol=pooled_physical})' 2>&1 | FileCheck %s --check-prefix=POOLED

module attributes {qlx.profiles = ["p2n"]} {
  lvm.domain @logical {
    lvm.space @exact_factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">],
      capacity = 1 : i64
    }
    lvm.space @pooled_factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">],
      capacity = 1 : i64
    }
    lvm.stream @exact_stream {
      backing_region = @exact_factory, produced_by = @exact_provider,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @exact_state
    }
    lvm.stream @pooled_stream {
      backing_region = @pooled_factory, produced_by = @pooled_provider,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @pooled_state
    }
    lvm.channel @exact_supply {
      from = @exact_factory, to = @exact_stream,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
    lvm.channel @pooled_supply {
      from = @pooled_factory, to = @pooled_stream,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @produce_exact : () -> !fabric.resource<@exact_state> {
    kind = "produce_exact_state"
  }
  qlx.action @produce_pooled : () -> !fabric.resource<@pooled_state> {
    kind = "produce_pooled_state"
  }
  fabric.protocol @exact_provider : () -> !fabric.resource<@exact_state>
      attributes {
        objective = @produce_exact,
        metadata = {acceptance_probability = "0.5",
                    cycles_per_attempt = "10", physical_qubits = "2",
                    factory_mode = "scheduled_macro", pipeline_depth = "1"}
      } {
    %state = fabric.produce_resource {
      region = @exact_factory, resource_kind = @exact_state
    } : !fabric.resource<@exact_state>
    fabric.protocol_return %state : !fabric.resource<@exact_state>
  }
  fabric.protocol @pooled_provider : () -> !fabric.resource<@pooled_state>
      attributes {
        objective = @produce_pooled,
        metadata = {acceptance_probability = "0.5",
                    cycles_per_attempt = "10", physical_qubits = "2",
                    factory_mode = "scheduled_macro", pipeline_depth = "1"}
      } {
    %state = fabric.produce_resource {
      region = @pooled_factory, resource_kind = @pooled_state
    } : !fabric.resource<@pooled_state>
    fabric.protocol_return %state : !fabric.resource<@pooled_state>
  }
  fabric.protocol @exact : () -> !event.handle<!fabric.resource<@exact_state>, "linear"> {
    %event = fabric.resource_request "exact_state" from @logical::@exact_stream
      : !event.handle<!fabric.resource<@exact_state>, "linear">
    fabric.protocol_return %event
      : !event.handle<!fabric.resource<@exact_state>, "linear">
  }
  fabric.protocol @pooled : () -> !event.handle<!fabric.resource<@pooled_state>, "linear"> {
    %event = fabric.resource_request "pooled_state" from @logical::@pooled_stream
      : !event.handle<!fabric.resource<@pooled_state>, "linear">
    fabric.protocol_return %event
      : !event.handle<!fabric.resource<@pooled_state>, "linear">
  }
  fabric.code @code {
    distance = 1 : i64, partitions = {data = 1 : i64}
  }
  fabric.machine @qec {
    fabric.region @exact_factory {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>
    }
    fabric.region @pooled_factory {
      code = @code, floorplan = #fabric.floorplan<direct, [1]>,
      role = #fabric.role<factory>
    }
  }
  phys.machine @arch {
    phys.resource_class @exact_qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
    phys.resource_class @pooled_qubits {
      kind = "qubit", count = 4 : i64, native_actions = []
    }
    phys.qec_binding @exact_binding {
      qec_region = @qec::@exact_factory, resources = [@exact_qubits]
    }
    phys.qec_binding @pooled_binding {
      qec_region = @qec::@pooled_factory, resources = [@pooled_qubits]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = 1000.0 : f64}
  }
  qlx.logical_to_qec @logical_to_qec {
    logical = @logical, qec = @qec,
    entries = [{logical = "exact_factory", qec = "exact_factory"},
               {logical = "pooled_factory", qec = "pooled_factory"}]
  }
  qlx.qec_to_physical @qec_to_physical {
    qec = @qec, physical = @arch,
    entries = [{qec = "exact_factory", binding = "exact_binding",
                resources = ["exact_qubits"]},
               {qec = "pooled_factory", binding = "pooled_binding",
                resources = ["pooled_qubits"]}]
  }
  qlx.device @device {
    logical = @logical, qec = @qec, physical = @arch,
    logical_to_qec = @logical_to_qec,
    qec_to_physical = @qec_to_physical,
    operating_point = @point
  }
}

// EXACT: phys.resource_request "exact_state" from @logical::@exact_stream
// EXACT-SAME: duration_ns = 2.000000e+04 : f64
// EXACT-SAME: factory_mode = "scheduled_macro"
// EXACT-SAME: physical_binding = @arch::@exact_binding

// POOLED: 'fabric.resource_request' op scheduled resource provider physical binding has 4 qubits but requires 2 qubits (exactly one engine footprint); pooled or multi-engine homes are unsupported
// POOLED-NOT: phys.graph @pooled_physical
