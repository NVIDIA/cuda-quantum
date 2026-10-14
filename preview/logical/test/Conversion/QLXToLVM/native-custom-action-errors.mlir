// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --qlx-to-lvm='root=portable domain=vm' 2>&1 | FileCheck %s
// RUN: not qlx-opt %s --qlx-to-lvm='root=destructive domain=vm' 2>&1 | FileCheck %s --check-prefix=DESTRUCTIVE

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_compute">],
      capacity = 1 : i64
    }
  }
  qlx.action @fresh : () -> !qlx.logical_qubit {kind = "custom"}
  qlx.action @destroy : (!qlx.logical_qubit) -> () {kind = "custom"}
  qlx.program @portable : () -> () attributes {qlx.stage = "p0"} {
    %q = qlx.apply @fresh() : () -> !qlx.logical_qubit
    qlx.discard %q : !qlx.logical_qubit
    qlx.return
  }
  qlx.program @destructive : (!qlx.logical_qubit) -> () attributes {qlx.stage = "p0"} {
  ^bb0(%q: !qlx.logical_qubit):
    qlx.apply @destroy(%q) : (!qlx.logical_qubit) -> ()
    qlx.return
  }
}

// CHECK: native placement requires a typed ownership map for custom quantum actions
// DESTRUCTIVE: native placement requires a typed ownership map for custom quantum actions
