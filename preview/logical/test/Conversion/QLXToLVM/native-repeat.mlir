// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --qlx-to-lvm | FileCheck %s

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">,
                      #lvm.capability<"qlx.machine/logical_reset">],
      capacity = 1 : i64
    }
  }
  qlx.program @folded : () -> i1 attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" : !qlx.logical_qubit
    %q1 = qlx.repeat 4
        iter(%q : !qlx.logical_qubit = %q0) {
      %next = qlx.apply #qlx.action<h>(%q)
        : (!qlx.logical_qubit) -> !qlx.logical_qubit
      qlx.yield %next : !qlx.logical_qubit
    }
    %m = qlx.measure <Z> %q1 : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK: lvm.kernel @folded_placed
// CHECK: %[[Q0:.*]] = lvm.prepare
// CHECK: %[[Q1:.*]] = "lvm.repeat"(%[[Q0]]) <{count = 4 : i64}>
// CHECK: lvm.yield
// CHECK: %[[M:.*]] = lvm.measure <Z> %[[Q1]]
// CHECK: lvm.return %[[M]] : i1
