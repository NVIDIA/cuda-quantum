// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["common"]} {
  qlx.lowering_recipe @stim_sample {
    accepted_stages = ["p2", "p3"],
    capability = "sample",
    effect = "local",
    finalizer = "qlx/stim-sample@0.3",
    plugin = {entry_point = "qlx.providers:stim", version = "0.1"},
    provides_facets = [],
    required_facets = ["qec_realization"],
    result_schema = "qlx.sample",
    stages = ["ensure-fabric-or-physical"]
  }
  qlx.lowering_recipe @stim_emit {
    accepted_stages = ["p2", "p3"],
    capability = "emit_text",
    effect = "local",
    finalizer = "qlx/stim-text@0.3",
    plugin = {entry_point = "qlx.providers:stim", version = "0.1"},
    provides_facets = [],
    required_facets = ["qec_realization"],
    result_schema = "text/stim-circuit",
    stages = ["ensure-fabric-or-physical"]
  }
  qlx.target_manifest @stim {
    availability = "local_optional_runtime",
    capabilities = ["sample", "emit_text"],
    plugin = {entry_point = "qlx.providers:stim", version = "0.1"},
    recipes = [@stim_sample, @stim_emit]
  }
}

// CHECK: qlx.lowering_recipe @stim_sample
// CHECK: qlx.lowering_recipe @stim_emit
// CHECK: qlx.target_manifest @stim
// CHECK-SAME: capabilities = ["sample", "emit_text"]
