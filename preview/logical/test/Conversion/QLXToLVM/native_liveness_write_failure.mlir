// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, system-linux
// A write to /dev/full fails with ENOSPC. There is no such device on macOS.
// RUN: not qlx-opt %s --qlx-to-lvm='root=sequential domain=vm result=write_failure witness-output=/dev/full' 2>&1 | FileCheck %s

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">,
                      #lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">],
      capacity = 1 : i64
    }
  }
  qlx.program @sequential : () -> (i1, i1) attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" : !qlx.logical_qubit
    %q1 = qlx.apply #qlx.action<h>(%q0)
      : (!qlx.logical_qubit) -> !qlx.logical_qubit
    %m0 = qlx.measure <Z> %q1 : !qlx.logical_qubit -> i1
    %q2 = qlx.prepare "zero" : !qlx.logical_qubit
    %m1 = qlx.measure <Z> %q2 : !qlx.logical_qubit -> i1
    qlx.return %m0, %m1 : i1, i1
  }
}

// CHECK: failed to write detached placement witness '/dev/full'
