// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @machine {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">,
                      #lvm.capability<"qlx.machine/logical_reset">],
      capacity = 2 : i64
    }
  }

  qlx.program @bell : () -> (i1, i1) attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" {allocation = 0 : i64, value_index = 0 : i64}
      : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" {allocation = 1 : i64, value_index = 1 : i64}
      : !qlx.logical_qubit
    %h = qlx.apply #qlx.action<h>(%q0)
      : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %c0, %c1 = qlx.apply #qlx.action<cx>(%h, %q1)
      : (!qlx.logical_qubit, !qlx.logical_qubit)
        -> (!qlx.logical_qubit, !qlx.logical_qubit)
    %m0 = qlx.measure <Z> %c0 : !qlx.logical_qubit -> i1
    %m1 = qlx.measure <Z> %c1 : !qlx.logical_qubit -> i1
    qlx.return %m0, %m1 : i1, i1
  }
}
