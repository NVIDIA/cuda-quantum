// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-translate
// RUN: not qlx-translate --fabric-to-stim %s 2>&1 | FileCheck %s

fabric.code @tiny {
  distance = 1 : i64,
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @tiny,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<direct, [1]>
  }
}

// CHECK: error: fabric-to-stim: unresolved executable call @missing
fabric.gadget @entry {entry} on @dev() {
  fabric.call @missing() : () -> ()
  fabric.return
}
