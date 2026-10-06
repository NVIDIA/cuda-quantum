// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-count{root=memory device=device result=static})' | FileCheck %s

fabric.code @steane {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

fabric.machine @qec {
  fabric.region @compute {
    code = @steane,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

fabric.gadget @memory {entry} on @qec() {
  %patch = fabric.alloc {code = @steane, region = @compute}
      : !fabric.patch<@steane>
  %out = fabric.h %patch data : !fabric.patch<@steane>
  %idle = fabric.idle %out {rounds = 3 : i64} : !fabric.patch<@steane>
  fabric.dealloc %idle : !fabric.patch<@steane>
  fabric.return
}

fabric.patch_graph @memory_graph {
  root = @memory,
  nodes = [{id = "patch0"}],
  interactions = []
}

lvm.domain @logical {
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
}

qlx.logical_to_qec @logical_to_qec {
  logical = @logical, qec = @qec,
  entries = [{logical = "compute", qec = "compute"}]
}

qlx.device @device {
  logical = @logical, qec = @qec,
  logical_to_qec = @logical_to_qec
}

// CHECK: qlx.estimate_result @static
// CHECK-SAME: patches_peak = 1 : i64
// CHECK-SAME: source_facets = ["qec_spec", "qec_realization", "patch_graph"]
// CHECK-SAME: device = @device
// CHECK-SAME: root = @memory
// CHECK-SAME: schema = "qlx.fabric-counts/v1"
// CHECK-SAME: tier = "static"
