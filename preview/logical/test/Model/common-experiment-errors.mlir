// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  qlx.program @memory : () -> () {
    qlx.return
  }
  // expected-error @+1 {{closure must contain the selected root}}
  qlx.experiment @missing_root {
    bindings = {}, closure = [], pass_recipe = [],
    stage = "p0", facets = [], root = @memory
  }
}

// -----

module {
  qlx.program @memory : () -> () {
    qlx.return
  }
  // expected-error @+1 {{pass_recipe entries require nonempty name and dictionary options}}
  qlx.experiment @bad_recipe {
    bindings = {}, closure = [@memory],
    pass_recipe = [{name = "", options = {}}],
    stage = "p0", facets = [], root = @memory
  }
}

// -----

module {
  qlx.program @memory : () -> () {
    qlx.return
  }
  // expected-error @+1 {{stage must be one of}}
  qlx.experiment @bad_stage {
    bindings = {}, closure = [@memory], pass_recipe = [],
    stage = "invalid", facets = [], root = @memory
  }
}
