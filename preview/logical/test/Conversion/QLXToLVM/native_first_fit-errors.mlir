// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --qlx-to-lvm 2>&1 | FileCheck %s

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">],
      capacity = 0 : i64
    }
  }
  qlx.program @portable : () -> () attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" : !qlx.logical_qubit
    qlx.discard %q0 : !qlx.logical_qubit
    qlx.return
  }
}

// CHECK: native P0-to-P1 placement found no logical space satisfying required capabilities and peak-live capacity in lvm.domain @vm
