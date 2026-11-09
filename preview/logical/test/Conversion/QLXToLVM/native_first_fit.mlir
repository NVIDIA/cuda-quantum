// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --qlx-to-lvm='root=portable domain=vm result=native' | FileCheck %s

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">,
                      #lvm.capability<"qlx.machine/logical_reset">],
      capacity = 2 : i64
    }
  }
  qlx.program @portable : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
      : !qlx.logical_qubit
    %q1 = qlx.apply #qlx.action<h>(%q0)
      : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m = qlx.measure <Z> %q1 : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK: lvm.kernel @native on @vm : () -> i1
// CHECK-SAME: input_p0 = @portable
// CHECK-SAME: placement_witness_sha256 = "sha256:
// CHECK-SAME: qlx.placement_policy = "native-first-fit/v3"
// CHECK: %[[Q0:.*]] = lvm.prepare "zero" at @vm::@compute
// CHECK: %[[Q1:.*]] = lvm.apply #qlx.action<h>(%[[Q0]]) at [@vm::@compute]
// CHECK: %[[M:.*]] = lvm.measure <Z> %[[Q1]] at @vm::@compute
// CHECK: lvm.return %[[M]] : i1
