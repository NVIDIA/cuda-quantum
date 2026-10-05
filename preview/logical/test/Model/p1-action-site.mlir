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
    lvm.action_site @generated_h {
      kind = "action",
      objective = #qlx.action<h>,
      placements = [@machine::@compute],
      source_site = 7 : i64
    }
  }

  lvm.kernel @placed on @machine
      : (!lvm.logical_qubit<@machine::@compute>)
     -> !lvm.logical_qubit<@machine::@compute> {
  ^bb0(%q: !lvm.logical_qubit<@machine::@compute>):
    %next = lvm.apply #qlx.action<h>(%q) at [@machine::@compute]
        {site = 7 : i64}
        : (!lvm.logical_qubit<@machine::@compute>)
       -> !lvm.logical_qubit<@machine::@compute>
    lvm.return %next : !lvm.logical_qubit<@machine::@compute>
  }
}

// CHECK: lvm.action_site @mpp
// CHECK-SAME: kind = "instrument"
// CHECK-SAME: objective = @logical_zz
// CHECK-SAME: placements = [@machine::@compute, @machine::@compute]
// CHECK: lvm.action_site @generated_h
// CHECK-SAME: kind = "action"
// CHECK-SAME: objective = #qlx.action<h>
// CHECK-SAME: placements = [@machine::@compute]
// CHECK-SAME: source_site = 7 : i64
// CHECK-NOT: selected
// CHECK-NOT: feasible_candidates
