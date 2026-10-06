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
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  qlx.program @portable : () -> i1 attributes {qlx.stage = "p0"} {
    %q = qlx.prepare "zero" : !qlx.logical_qubit
    %m = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
    qlx.return %m : i1
  }
}

// CHECK: no logical space satisfying required capabilities and peak-live capacity
