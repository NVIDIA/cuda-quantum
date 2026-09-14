// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.code @source {
    distance = 1 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64
  }
  fabric.code_profile @source_profile {
    code = @source, distance_status = "claimed"
  }
  fabric.encoding @source_encoding {
    block = "block0", code = @source,
    logical_ports = ["q0"], profile = @source_profile
  }
  fabric.code @destination {
    distance = 1 : i64, k = 1 : i64, n = 2 : i64,
    partitions = {data = 2 : i64}, r = 0 : i64
  }
  fabric.code_profile @destination_profile {
    code = @destination, distance_status = "claimed"
  }
  fabric.encoding @destination_encoding {
    block = "block0", code = @destination,
    logical_ports = ["q0"], profile = @destination_profile
  }

  // expected-error @+1 {{'fabric.patch_transform' op source_support width must equal the referenced code n (1)}}
  fabric.patch_transform @bad_width
      from @source_encoding to @destination_encoding {
    destination_roles = {
      active = array<i64: 0, 1>, dormant = array<i64>,
      measured = array<i64>, reset = array<i64>, scratch = array<i64>
    },
    destination_support = array<i64: 0, 1>,
    evidence = "invalid_width_test",
    frame_partitions = {data = 2 : i64},
    logical_map = array<i64: 0>,
    source_roles = {
      active = array<i64: 0>, dormant = array<i64: 1>,
      measured = array<i64>, reset = array<i64>, scratch = array<i64>
    },
    source_support = array<i64: 0, 1>
  }
}

// -----

module {
  func.func @bad_bits(%arg0: tensor<2xi8>) -> i1 {
    // expected-error @+1 {{'fabric.all_zero' op requires a tensor<Nxi1> measurement bundle}}
    %accepted = fabric.all_zero %arg0 : tensor<2xi8> -> i1
    return %accepted : i1
  }
}

// -----

module {
  fabric.code @code {
    distance = 1 : i64, k = 1 : i64, n = 1 : i64,
    partitions = {data = 1 : i64}, r = 0 : i64
  }
  fabric.code_profile @profile {
    code = @code, distance_status = "claimed"
  }
  fabric.encoding @encoding {
    block = "block0", code = @code,
    logical_ports = ["q0"], profile = @profile
  }

  // expected-error @+1 {{'fabric.patch_transform' op source_roles active role must equal the boundary support}}
  fabric.patch_transform @bad_roles from @encoding to @encoding {
    destination_roles = {
      active = array<i64: 0>, dormant = array<i64>,
      measured = array<i64>, reset = array<i64>, scratch = array<i64>
    },
    destination_support = array<i64: 0>,
    evidence = "invalid_roles_test",
    frame_partitions = {data = 1 : i64},
    logical_map = array<i64: 0>,
    source_roles = {
      active = array<i64>, dormant = array<i64: 0>,
      measured = array<i64>, reset = array<i64>, scratch = array<i64>
    },
    source_support = array<i64: 0>
  }
}
