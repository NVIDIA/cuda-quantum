// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

fabric.code @bare {
  distance = 1 : i64,
  n = 1 : i64,
  k = 1 : i64,
  r = 0 : i64,
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @qec {
  fabric.region @memory {
    code = @bare,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<memory>
  }
}

fabric.gadget @memory {entry} on @qec() -> tensor<1xi1> {
  %patch = fabric.alloc {code = @bare, region = @memory}
      : !fabric.patch<@bare>
  %prepared = fabric.prep_z %patch : !fabric.patch<@bare>
  %out, %bits = fabric.mz %prepared data
      : !fabric.patch<@bare> -> tensor<1xi1>
  fabric.dealloc %out : !fabric.patch<@bare>
  fabric.return %bits : tensor<1xi1>
}
