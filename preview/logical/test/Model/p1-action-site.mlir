// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {
  qlx.model_version = "0.3.10-proposed",
  qlx.profiles = ["p1"]
} {
  lvm.domain @machine {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/lattice_surgery">],
      capacity = 2 : i64
    }
    lvm.action_site @mpp {
      kind = "instrument",
      objective = @logical_zz,
      placements = [@machine::@compute, @machine::@compute]
    }
  }
}

// CHECK: lvm.action_site @mpp
// CHECK-SAME: kind = "instrument"
// CHECK-SAME: objective = @logical_zz
// CHECK-SAME: placements = [@machine::@compute, @machine::@compute]
// CHECK-NOT: selected
// CHECK-NOT: feasible_candidates
