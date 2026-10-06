// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

// Native product measurement is legal only when every participating
// resource class advertises the typed instrument in native_instruments
// (spec 05 "measure_product", conformance case 83). A bare "mpp" entry in
// native_actions does not satisfy the gate.

module {
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
      native_actions = ["mpp"]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @no_native_mpp on @arch : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@q0, @q1] : !phys.state<@q0>, !phys.state<@q1>
    // expected-error @+1 {{resource class @qubits of state operand 0 does not advertise native instrument @mpp}}
    %1:3 = phys.measure_product %0#0, %0#1 {
      instrument = @mpp,
      paulis = ["X", "Z"],
      record_id = "mpp.outcome"
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    phys.return %1#2 : !phys.record<@bit>
  }
}

// -----

// Advertising a different typed instrument does not cover @mpp.

module {
  phys.instrument @mpp {
    kind = "measure_product",
    variadic,
    record_schema = "bit",
    preserves_inputs,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  phys.instrument @other_mpp {
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
      native_actions = ["rpp"],
      native_instruments = [@other_mpp]
    }
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @q1 {index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @wrong_native_mpp on @arch : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@q0, @q1] : !phys.state<@q0>, !phys.state<@q1>
    // expected-error @+1 {{resource class @qubits of state operand 0 does not advertise native instrument @mpp}}
    %1:3 = phys.measure_product %0#0, %0#1 {
      instrument = @mpp,
      paulis = ["X", "Z"],
      record_id = "mpp.outcome"
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    phys.return %1#2 : !phys.record<@bit>
  }
}

// -----

// P3 ownership fails closed even for partially linked IR: every terminal
// state must resolve to a concrete phys.resource.

module {
  phys.instrument @mpp {
    kind = "measure_product",
    variadic,
    record_schema = "bit",
    preserves_inputs,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  // expected-error @+1 {{architecture must resolve to phys.machine}}
  phys.graph @partial on @missing_arch : (!phys.state<@undeclared_q>)
      -> !phys.record<@bit> {
  ^bb0(%q: !phys.state<@undeclared_q>):
    %0:2 = phys.measure_product %q {
      instrument = @mpp,
      paulis = ["Z"],
      record_id = "mpp.outcome"
    } : (!phys.state<@undeclared_q>) -> (!phys.state<@undeclared_q>, !phys.record<@bit>)
    phys.release %0#0 : !phys.state<@undeclared_q>
    phys.return %0#1 : !phys.record<@bit>
  }
}
