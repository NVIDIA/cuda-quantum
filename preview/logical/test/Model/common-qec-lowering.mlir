// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p0", "p2s"]} {
  fabric.code @surface {
    distance = 5 : i64,
    k = 1 : i64,
    n = 25 : i64,
    partitions = {data = 25 : i64, sx = 12 : i64, sz = 12 : i64},
    r = 0 : i64
  }
  qlx.qec_lowering @surface_mpp {
    manifest_name = "surface_mpp",
    manifest_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    codes = [@surface],
    compiler_plugin = "qlx.lattice_surgery",
    compiler_symbol = "surface_mpp",
    compiler_version = "1.0.0",
    dependencies = [@prepare, @merge, @measure, @split],
    input_stage = "p1",
    objective = #qlx.instrument<mpp>,
    objective_family = "pauli_product_measurement",
    output_stage = "p2",
    provides_facets = ["qec_realization", "protocol_network"],
    policy_schema = {max_weight = "int"},
    requirements = ["qlx.machine/lattice_surgery"]
  }
}

// CHECK: qlx.qec_lowering @surface_mpp
// CHECK-SAME: compiler_plugin = "qlx.lattice_surgery"
// CHECK-SAME: objective_family = "pauli_product_measurement"
// CHECK-SAME: requirements = ["qlx.machine/lattice_surgery"]
