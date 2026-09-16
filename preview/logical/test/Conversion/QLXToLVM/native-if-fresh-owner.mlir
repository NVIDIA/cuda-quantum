// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --qlx-to-lvm='root=fresh_join domain=vm result=placed' | FileCheck %s

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">,
                      #lvm.capability<"qlx.machine/logical_measurement">],
      capacity = 1 : i64
    }
  }
  qlx.program @fresh_join : (i1) -> i1 attributes {qlx.stage = "p0"} {
  ^bb0(%condition: i1):
    %q = "cflow.if"(%condition) ({
      %then = qlx.prepare "zero" : !qlx.logical_qubit
      cflow.yield %then : !qlx.logical_qubit
    }, {
      %else = qlx.prepare "zero" : !qlx.logical_qubit
      cflow.yield %else : !qlx.logical_qubit
    }) : (i1) -> !qlx.logical_qubit
    %measured = qlx.measure <Z> %q : !qlx.logical_qubit -> i1
    qlx.return %measured : i1
  }
}

// CHECK: %[[Q:.*]] = cflow.if
// CHECK: lvm.prepare "zero" at @vm::@compute
// CHECK: lvm.prepare "zero" at @vm::@compute
// CHECK: lvm.measure
