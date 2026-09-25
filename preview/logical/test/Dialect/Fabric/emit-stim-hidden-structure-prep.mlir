// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: not qlx-translate --fabric-to-stim %s 2>&1 | FileCheck %s

fabric.code @encoded {
  distance = 1 : i64,
  n = 1 : i64,
  k = 1 : i64,
  r = 0 : i64,
  stabilizers = ["Z"],
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @encoded,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @entry {entry} on @dev() {
  %patch = fabric.alloc {code = @encoded, region = @C0}
      : !fabric.patch<@encoded>
  // CHECK: error: fabric-to-stim: prep_x/prep_z require a proven trivial one-carrier code
  %prepared = fabric.prep_z %patch : !fabric.patch<@encoded>
  fabric.dealloc %prepared : !fabric.patch<@encoded>
  fabric.return
}
