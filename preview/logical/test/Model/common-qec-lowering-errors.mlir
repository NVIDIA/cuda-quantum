// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --split-input-file --verify-diagnostics %s

module {
  fabric.code @surface {
    distance = 3 : i64,
    partitions = {data = 1 : i64}
  }
  // expected-error @+1 {{requires attribute 'manifest_name'}}
  qlx.qec_lowering @missing_manifest_name {
    manifest_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    codes = [@surface],
    compiler_plugin = "qlx.test",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [],
    input_stage = "p1",
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    provides_facets = ["qec_realization", "protocol_network"],
    requirements = []
  }
}

// -----

module {
  // expected-error @+1 {{accepted code or encoding @missing_code does not resolve}}
  qlx.qec_lowering @unresolved_code {
    manifest_name = "unresolved_code",
    manifest_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    codes = [@missing_code],
    compiler_plugin = "qlx.test",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [],
    input_stage = "p1",
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    provides_facets = ["qec_realization", "protocol_network"],
    requirements = []
  }
}

// -----

module {
  fabric.code @surface {
    distance = 3 : i64,
    partitions = {data = 1 : i64}
  }
  // expected-error @+1 {{policy_schema must contain nonempty keys and string values}}
  qlx.qec_lowering @noncanonical_policy {
    manifest_name = "noncanonical_policy",
    manifest_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    codes = [@surface],
    compiler_plugin = "qlx.test",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [],
    input_stage = "p1",
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    policy_schema = {weight = 3 : i64},
    provides_facets = ["qec_realization", "protocol_network"],
    requirements = []
  }
}

// -----

module {
  fabric.code @surface {
    distance = 3 : i64,
    partitions = {data = 1 : i64}
  }
  // expected-error @+1 {{metadata must contain nonempty keys and string values}}
  qlx.qec_lowering @noncanonical_metadata {
    manifest_name = "noncanonical_metadata",
    manifest_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    codes = [@surface],
    compiler_plugin = "qlx.test",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [],
    input_stage = "p1",
    metadata = {enabled = true},
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    provides_facets = ["qec_realization", "protocol_network"],
    requirements = []
  }
}

// -----

module {
  fabric.code @surface {
    distance = 3 : i64,
    partitions = {data = 1 : i64}
  }
  // expected-error @+1 {{manifest_name must be nonempty}}
  qlx.qec_lowering @empty_manifest_name {
    manifest_name = "",
    manifest_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    codes = [@surface],
    compiler_plugin = "qlx.test",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [],
    input_stage = "p1",
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    provides_facets = ["qec_realization", "protocol_network"],
    requirements = []
  }
}

// -----

module {
  fabric.code @surface {
    distance = 3 : i64,
    partitions = {data = 1 : i64}
  }
  // expected-error @+1 {{manifest_sha256 must be sha256: followed by 64 lowercase hexadecimal digits}}
  qlx.qec_lowering @malformed_manifest_digest {
    manifest_name = "malformed_manifest_digest",
    manifest_sha256 = "sha256:ABCDEF",
    codes = [@surface],
    compiler_plugin = "qlx.test",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [],
    input_stage = "p1",
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    provides_facets = ["qec_realization", "protocol_network"],
    requirements = []
  }
}
