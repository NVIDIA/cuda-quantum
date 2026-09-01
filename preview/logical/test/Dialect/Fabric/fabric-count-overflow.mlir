// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt --fabric-count %s 2>&1 | FileCheck %s

fabric.code @steane {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

fabric.machine @dev {
  fabric.region @C0 {
    code = @steane,
    floorplan = #fabric.floorplan<direct, [4]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @overflow {entry} on @dev() {
  %p = fabric.alloc {code = @steane, region = @C0} : !fabric.patch<@steane>
  %out = fabric.repeat 9223372036854775807 iter(%arg : !fabric.patch<@steane> = %p) {
    %one = fabric.h %arg data : !fabric.patch<@steane>
    %two = fabric.h %one data : !fabric.patch<@steane>
    fabric.yield %two : !fabric.patch<@steane>
  }
  fabric.dealloc %out : !fabric.patch<@steane>
  fabric.return
}

// CHECK: fabric-count gate count overflows signed i64
