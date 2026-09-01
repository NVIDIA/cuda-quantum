// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --fabric-count='device=stack' 2>&1 | FileCheck %s

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

lvm.domain @logical {
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
}

qlx.logical_to_qec @map {
  logical = @logical,
  qec = @dev,
  entries = [{logical = "compute", qec = "C0"}]
}

qlx.device @stack {logical = @logical, qec = @dev, logical_to_qec = @map}

fabric.gadget @recursive {entry} on @dev() {
  fabric.call @recursive() : () -> ()
  fabric.return
}

// CHECK: fabric-count rejects recursive executable call through @recursive
