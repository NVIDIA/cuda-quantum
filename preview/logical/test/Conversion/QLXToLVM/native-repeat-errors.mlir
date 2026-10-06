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
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">,
                      #lvm.capability<"qlx.machine/logical_measurement">],
      capacity = 2 : i64
    }
  }
  qlx.program @permuted : () -> (i1, i1) attributes {qlx.stage = "p0"} {
    %left = qlx.prepare "zero" : !qlx.logical_qubit
    %right = qlx.prepare "zero" : !qlx.logical_qubit
    %out_left, %out_right = cflow.repeat 2
        iter(%a : !qlx.logical_qubit = %left,
             %b : !qlx.logical_qubit = %right) {
      cflow.yield %b, %a : !qlx.logical_qubit, !qlx.logical_qubit
    }
    %m0 = qlx.measure <Z> %out_left : !qlx.logical_qubit -> i1
    %m1 = qlx.measure <Z> %out_right : !qlx.logical_qubit -> i1
    qlx.return %m0, %m1 : i1, i1
  }
}

// CHECK: native placement requires each repeat yield to preserve its carried owner's exact logical slot
