// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

fabric.code @steane {
  distance = 3 : i64,
  metadata = {distance_method = "fixture",
              distance_provenance = @steane_evidence},
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

fabric.code_profile @steane_evidence {
  code = @steane,
  distance_claim = 3 : i64,
  distance_status = "exact",
  evidence = ["qlx-cli-example@1"]
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
