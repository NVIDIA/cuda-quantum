// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.instrument @mpp {
    kind = "measure_product",
    variadic,
    record_schema = "bit",
    preserves_inputs,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64,
      kind = "qubit",
      native_actions = [],
      native_instruments = [@mpp]
    }
  }
  phys.resource @q0 {
    capabilities = ["qlx.physical/native_pauli_product_rotation"],
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    capabilities = ["qlx.physical/native_pauli_product_rotation"],
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @mpp_graph on @arch : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire.mpp"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:3 = phys.measure_product %0#0, %0#1 {
      event_id = "mpp0",
      instrument = @mpp,
      paulis = ["X", "Z"],
      record_id = "mpp.outcome"
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    phys.release %1#0, %1#1 {event_id = "release.mpp"} :
      !phys.state<@q0>, !phys.state<@q1>
    phys.return %1#2 : !phys.record<@bit>
  }
  phys.graph @negated_mpp_graph on @arch : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire.negated"}
      : !phys.state<@q0>, !phys.state<@q1>
    // A Fabric-level negated product (pauli_product = "-...") projects to
    // the `invert` unit attribute: the record is the complement of the
    // unsigned product's outcome (Stim's `MPP !P0*P1`).
    %1:3 = phys.measure_product %0#0, %0#1 {
      event_id = "mpp1",
      instrument = @mpp,
      invert,
      paulis = ["Z", "Z"],
      record_id = "neg_mpp.outcome"
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    phys.release %1#0, %1#1 {event_id = "release.negated"} :
      !phys.state<@q0>, !phys.state<@q1>
    phys.return %1#2 : !phys.record<@bit>
  }
  phys.graph @rpp on @arch : () -> () {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "acquire.rpp"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = phys.rotate_product %0#0, %0#1 {
      angle = 1.250000e-01 : f64,
      event_id = "rpp0",
      paulis = ["Y", "Z"]
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>)
    phys.release %1#0, %1#1 {event_id = "release.rpp"} :
      !phys.state<@q0>, !phys.state<@q1>
    phys.return
  }
}

// CHECK: phys.measure_product
// CHECK-SAME: instrument = @mpp
// CHECK-SAME: paulis = ["X", "Z"]
// CHECK-SAME: record_id = "mpp.outcome"
// CHECK: phys.measure_product
// CHECK-SAME: invert
// CHECK-SAME: paulis = ["Z", "Z"]
// CHECK-SAME: record_id = "neg_mpp.outcome"
// CHECK: phys.rotate_product
// CHECK-SAME: angle = 1.250000e-01 : f64
// CHECK-SAME: paulis = ["Y", "Z"]
