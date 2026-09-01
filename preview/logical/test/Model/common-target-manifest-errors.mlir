// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  qlx.lowering_recipe @emit {
    accepted_stages = ["p2"], required_facets = [], provides_facets = [],
    capability = "emit_text", effect = "local", finalizer = "emit", stages = []
  }
  // expected-error @+1 {{contains duplicate capability 'emit_text'}}
  qlx.target_manifest @duplicate {
    availability = "local", capabilities = ["emit_text", "emit_text"],
    recipes = [@emit]
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
