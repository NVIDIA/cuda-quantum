// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: not qlx-translate --fabric-to-stim %s 2>&1 | FileCheck %s

fabric.code @hadamard_encoded {
  anti_stabilizers = dense<> : tensor<0x2xi1>,
  distance = 1 : i64,
  encoding_clifford = dense<[[false, true], [true, false]]> : tensor<2x2xi1>,
  gauge_x_basis = dense<> : tensor<0x2xi1>,
  gauge_z_basis = dense<> : tensor<0x2xi1>,
  k = 1 : i64,
  logical_x_basis = dense<[[false, true]]> : tensor<1x2xi1>,
  logical_z_basis = dense<[[true, false]]> : tensor<1x2xi1>,
  lx = [array<i64: 0>],
  lz = [array<i64: 0>],
  n = 1 : i64,
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64},
  r = 0 : i64,
  stabilizer_basis = dense<> : tensor<0x2xi1>
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @hadamard_encoded,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @entry {entry} on @dev() {
  %patch = fabric.alloc {code = @hadamard_encoded, region = @C0}
      : !fabric.patch<@hadamard_encoded>
  // CHECK: error: fabric-to-stim: prep_x/prep_z require a proven trivial one-carrier code
  %prepared = fabric.prep_z %patch : !fabric.patch<@hadamard_encoded>
  fabric.dealloc %prepared : !fabric.patch<@hadamard_encoded>
  fabric.return
}
