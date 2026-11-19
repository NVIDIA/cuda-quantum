// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @machine {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">],
      capacity = 1 : i64
    }
  }
  lvm.kernel @control on @machine : () -> () attributes {qlx.profile = "p1"} {
    %q = lvm.prepare "zero" at @machine::@compute : !lvm.logical_qubit<@machine::@compute>
    %condition = arith.constant true
    %selected = "cflow.if"(%condition) ({
      cflow.yield %q : !lvm.logical_qubit<@machine::@compute>
    }, {
      cflow.yield %q : !lvm.logical_qubit<@machine::@compute>
    }) : (i1) -> !lvm.logical_qubit<@machine::@compute>
    %repeated = "cflow.repeat"(%selected) <{count = 4 : i64}> ({
    ^bb0(%iter: !lvm.logical_qubit<@machine::@compute>):
      cflow.yield %iter : !lvm.logical_qubit<@machine::@compute>
    }) : (!lvm.logical_qubit<@machine::@compute>) -> !lvm.logical_qubit<@machine::@compute>
    lvm.discard %repeated at [@machine::@compute] : !lvm.logical_qubit<@machine::@compute>
    lvm.return
  }
}

// CHECK: cflow.if
// CHECK: cflow.repeat 4 iter
// CHECK: cflow.yield
