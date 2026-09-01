// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-count{root=memory_protocol device=device result=static})' | FileCheck %s

fabric.code @tiny {
  distance = 1 : i64,
  partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}
}

fabric.machine @qec {
  fabric.region @compute {
    code = @tiny,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<compute>
  }
}

// Canonical compiler-produced protocols authenticate the selected QEC choice
// with the retained P1 input and its selection commitment. They do not need an
// incidental device annotation on a called gadget.
fabric.protocol @memory_protocol : () -> () attributes {
  metadata = {
    input_p1 = "placed_kernel",
    qec_selection_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  }
} {
  %patch = fabric.alloc {code = @tiny, region = @compute}
      : !fabric.patch<@tiny>
  fabric.dealloc %patch : !fabric.patch<@tiny>
  fabric.protocol_return
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
// CHECK-SAME: logical_qubits_peak = 1 : i64
// CHECK-SAME: source_facets = ["qec_spec", "protocol_network"]
// CHECK-SAME: device = @device
// CHECK-SAME: root = @memory_protocol
