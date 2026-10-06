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
  qlx.profiles = ["p0", "p1"]
} {
  lvm.domain @vm attributes {qlx.profile = "p1"} {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">,
                      #lvm.capability<"qlx.machine/logical_reset">],
      capacity = 2 : i64,
      tags = ["compute"]
    }
  }

  lvm.kernel @placed on @vm : () -> (i1) attributes {
    input_p0 = @portable,
    placement_witness_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    qlx.profile = "p1"
  } {
    %q0 = lvm.prepare "zero" at @vm::@compute
      : !lvm.logical_qubit<@vm::@compute>
    %q1 = lvm.apply #qlx.action<h>(%q0) at [@vm::@compute] {site = 0 : i64}
      : (!lvm.logical_qubit<@vm::@compute>)
        -> (!lvm.logical_qubit<@vm::@compute>)
    %m = lvm.measure <Z> %q1 at @vm::@compute {site = 1 : i64}
      : !lvm.logical_qubit<@vm::@compute> -> i1
    lvm.return %m : i1
  }

  qlx.program @portable : () -> (i1) attributes {qlx.profile = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64} : !qlx.logical_qubit
    %q1 = qlx.apply #qlx.action<h>(%q0) : (!qlx.logical_qubit) -> (!qlx.logical_qubit)
    %m = qlx.measure <Z> %q1 : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK: lvm.domain @vm
// CHECK: lvm.space @compute
// CHECK: lvm.kernel @placed on @vm : () -> i1
// CHECK: %[[Q0:.*]] = lvm.prepare "zero" at @vm::@compute : !lvm.logical_qubit<@vm::@compute>
// CHECK: %[[Q1:.*]] = lvm.apply #qlx.action<h>(%[[Q0]]) at [@vm::@compute]
// CHECK: %[[M:.*]] = lvm.measure <Z> %[[Q1]] at @vm::@compute
// CHECK: lvm.return %[[M]] : i1
