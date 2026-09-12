// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  func.func @bad(%q: !phys.state<@q>) {
    // expected-error @+1 {{requires one Pauli label per state operand}}
    %0:2 = phys.measure_product %q {
      instrument = @mpp,
      paulis = ["X", "Z"],
      record_id = "m"
    } : (!phys.state<@q>) -> (!phys.state<@q>, !phys.record<@bit>)
    return
  }
}

// -----

module {
  func.func @bad_rpp(%q: !phys.state<@q>) {
    // expected-error @+1 {{requires one Pauli label per state operand}}
    %0 = phys.rotate_product %q {
      angle = 2.500000e-01 : f64,
      paulis = ["X", "Z"]
    } : (!phys.state<@q>) -> !phys.state<@q>
    return
  }
}

// -----

module {
  func.func @bad_resource_rpp(
      %raw: !phys.resource_payload<@raw_t_state>,
      %q: !phys.state<@q>) {
    // expected-error @+1 {{requires one Pauli label per state operand}}
    %0 = phys.resource_rotate_product %raw on %q {
      angle = 7.8539816339744828e-01 : f64,
      paulis = ["X", "Z"]
    } : (!phys.resource_payload<@raw_t_state>, !phys.state<@q>)
      -> !phys.state<@q>
    return
  }
}

// -----

module {
  phys.instrument @unary_mpp {
    arity = 1 : i64,
    kind = "measure_product",
    preserves_inputs,
    record_schema = "bit",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  func.func @bad_instrument_arity(%q0: !phys.state<@q0>,
                                  %q1: !phys.state<@q1>) {
    // expected-error @+1 {{operand count does not match fixed instrument arity}}
    %0:3 = phys.measure_product %q0, %q1 {
      instrument = @unary_mpp,
      paulis = ["X", "Z"],
      record_id = "m"
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    return
  }
}

// -----

module {
  phys.instrument @destructive_mpp {
    kind = "measure_product",
    variadic,
    record_schema = "bit",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  func.func @bad_instrument_ownership(%q: !phys.state<@q>) {
    // expected-error @+1 {{measure_product instrument must preserve input states}}
    %0:2 = phys.measure_product %q {
      instrument = @destructive_mpp,
      paulis = ["X"],
      record_id = "m"
    } : (!phys.state<@q>) -> (!phys.state<@q>, !phys.record<@bit>)
    return
  }
}
