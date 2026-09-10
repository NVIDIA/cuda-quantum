// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  fabric.code @source_code {
    distance = 1 : i64, n = 1 : i64, k = 1 : i64, r = 0 : i64,
    partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}
  }
  fabric.gadget @source(%patch: !fabric.patch<@source_code>)
      -> (!fabric.patch<@source_code>, i1) {
    %next, %record = fabric.measure_product %patch {
      logical_indices = array<i64: 0>, patch_indices = array<i64: 0>,
      pauli_product = "Z", record = "final"
    } : (!fabric.patch<@source_code>) -> (!fabric.patch<@source_code>, i1)
    fabric.return %next, %record : !fabric.patch<@source_code>, i1
  }
  fabric.gadget_profile @analysis for @source {
    fabric.success {records = ["source.final.outcome"]}
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64,
      kind = "qubit",
      native_actions = ["measure_z"]
    }
  }
  phys.resource @q0 {
    index = 0 : i64,
    kind = "qubit",
    resource_class = @qubits
  }
  phys.graph @readout on @arch : () -> !phys.state<@q0> attributes {
    source_protocol = @source
  } {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %1:2 = phys.measure @measure_z(%0) {
      event_id = "m0",
      record_id = "readout.final.data0"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %1#0 : !phys.state<@q0>
  }
  phys.record_projection @projection for @readout from @source {
    entries = [{
      instance = "source",
      physical_record = "readout.final.data0",
      source_record = "source.final.outcome"
    }]
  }
  phys.selection_sidecar @success for @readout {
    constant = false,
    expected = false,
    input_syndromes = [],
    projection_indices = array<i64: 0>,
    record_projection = @projection,
    records = ["readout.final.data0"],
    source_kind = "profile",
    source_instance = "source",
    source_profile = @analysis,
    source_records = ["source.final.outcome"],
    source_row = 0 : i64
  }
}

// CHECK: phys.record_projection @projection for @readout from @source
// CHECK-SAME: source_record = "source.final.outcome"
// CHECK: phys.selection_sidecar @success for @readout
// CHECK-SAME: source_kind = "profile"
// CHECK-SAME: source_profile = @analysis
