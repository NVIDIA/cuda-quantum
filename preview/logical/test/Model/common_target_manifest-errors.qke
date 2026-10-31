// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  qlx.lowering_recipe @sample {
    accepted_stages = ["p3"], required_facets = [], provides_facets = [],
    capability = "sample", effect = "local", finalizer = "sample", stages = []
  }
  // expected-error @+1 {{capabilities must equal referenced recipe capabilities}}
  qlx.target_manifest @missing {
    availability = "local", capabilities = ["sample", "run"],
    recipes = [@sample]
  }
}

// -----

module {
  qlx.lowering_recipe @sample {
    accepted_stages = ["p3"], required_facets = [], provides_facets = [],
    capability = "sample", effect = "local", finalizer = "sample", stages = []
  }
  // expected-error @+1 {{contains duplicate capability 'sample'}}
  qlx.target_manifest @duplicate {
    availability = "local", capabilities = ["sample", "sample"],
    recipes = [@sample]
  }
}

// -----

module {
  // expected-error @+1 {{effect must be local, filesystem, or external}}
  qlx.lowering_recipe @bad_effect {
    accepted_stages = [], required_facets = [], provides_facets = [],
    capability = "emit_text", effect = "networkish", finalizer = "emit",
    stages = []
  }
}
