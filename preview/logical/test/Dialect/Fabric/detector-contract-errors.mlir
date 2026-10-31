// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under    //
// the terms of the Apache License 2.0 which accompanies this distribution.    //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

// Partial linked profiles still check declaration-local structure. Redundant
// and constant rows are valid declarations; ideal correctness is separate.
fabric.gadget_contract @partial for @later {
  fabric.detector
  fabric.detector {constant = true}
  fabric.observable {idx = 0 : i64}
  fabric.observable {constant = true, idx = 1 : i64}
}

// -----

// expected-error @+1 {{must be a direct child of a fabric.gadget_contract}}
fabric.detector

// -----

// expected-error @+1 {{must be a direct child of a fabric.gadget_contract}}
fabric.observable {idx = 0 : i64}

// -----

fabric.gadget_contract @partial for @later {
  // expected-error @+1 {{observable index must be nonnegative}}
  fabric.observable {idx = -1 : i64}
}

// -----

fabric.gadget_contract @partial for @later {
  // expected-error @+1 {{observable rows cannot use input virtual syndromes}}
  fabric.observable {
    idx = 0 : i64,
    input_syndromes = [{port_index = 0 : i64, index = 0 : i64}]
  }
}

// -----

// Nested declarations must not evade the profile's row checks.
fabric.gadget_contract @nested_detector for @later {
  builtin.module {
    // expected-error @+1 {{must be a direct child of a fabric.gadget_contract}}
    fabric.detector {constant = true}
  }
}

// -----

fabric.gadget_contract @nested_observable for @later {
  builtin.module {
    // expected-error @+1 {{must be a direct child of a fabric.gadget_contract}}
    fabric.observable {constant = true, idx = 99 : i64}
  }
}
