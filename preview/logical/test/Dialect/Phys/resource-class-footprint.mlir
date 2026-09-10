// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --split-input-file --verify-diagnostics %s

module {
  phys.machine @valid {
    phys.resource_class @patches {
      count = 4 : i64,
      footprint_evidence = "square distance-3 surface-code patch",
      granularity = "patch",
      kind = "surface_code_patch",
      native_actions = [],
      physical_unit_kind = "qubit",
      physical_units = 32 : i64
    }
  }
}

// -----

module {
  phys.machine @unknown {
    // expected-error @+1 {{granularity must be 'carrier' or 'patch'}}
    phys.resource_class @patches {
      count = 1 : i64, granularity = "block", kind = "patch",
      native_actions = []
    }
  }
}

// -----

module {
  phys.machine @missing {
    // expected-error @+1 {{patch granularity requires nonempty physical_unit_kind and footprint_evidence plus positive physical_units}}
    phys.resource_class @patches {
      count = 1 : i64, granularity = "patch", kind = "patch",
      native_actions = []
    }
  }
}

// -----

module {
  phys.machine @nonpositive {
    // expected-error @+1 {{patch granularity requires nonempty physical_unit_kind and footprint_evidence plus positive physical_units}}
    phys.resource_class @patches {
      count = 1 : i64, footprint_evidence = "invalid",
      granularity = "patch", kind = "patch", native_actions = [],
      physical_unit_kind = "qubit", physical_units = 0 : i64
    }
  }
}

// -----

module {
  phys.machine @carrier_footprint {
    // expected-error @+1 {{carrier granularity must not declare a patch physical footprint}}
    phys.resource_class @qubits {
      count = 1 : i64, footprint_evidence = "invalid",
      granularity = "carrier", kind = "qubit", native_actions = [],
      physical_unit_kind = "qubit", physical_units = 1 : i64
    }
  }
}
