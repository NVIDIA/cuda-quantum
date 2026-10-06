// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --qlx-to-lvm='root=sequential domain=vm result=placed' | FileCheck %s
// RUN: qlx-opt %s --qlx-to-lvm='root=sequential domain=vm result=with_witness witness-output=%t.json' > %t.mlir
// RUN: FileCheck %s --check-prefix=WITNESS < %t.json
// RUN: FileCheck %s --check-prefix=COMMIT < %t.mlir
// RUN: not qlx-opt %s --qlx-to-lvm='root=sequential domain=vm result=write_failure witness-output=/dev/full' 2>&1 | FileCheck %s --check-prefix=WRITE-FAILURE
// RUN: qlx-opt %s --qlx-to-lvm='root=non_lifo_reuse domain=vm_two result=reused witness-output=%t-reuse.json' > /dev/null
// RUN: FileCheck %s --check-prefix=REUSE < %t-reuse.json

module attributes {qlx.profiles = ["p0", "p1"]} {
  lvm.domain @vm {
    lvm.space @unsupported {capabilities = [], capacity = 4 : i64}
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">,
                      #lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">],
      capacity = 1 : i64
    }
  }
  lvm.domain @vm_two {
    lvm.space @compute {
      capabilities = [#lvm.capability<"qlx.machine/logical_reset">,
                      #lvm.capability<"qlx.machine/logical_compute">,
                      #lvm.capability<"qlx.machine/logical_measurement">],
      capacity = 2 : i64
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
  qlx.program @non_lifo_reuse : () -> (i1, i1) attributes {qlx.stage = "p0"} {
    %q0 = qlx.prepare "zero" : !qlx.logical_qubit
    %q1 = qlx.prepare "zero" : !qlx.logical_qubit
    %m0 = qlx.measure <Z> %q0 : !qlx.logical_qubit -> i1
    %q2 = qlx.prepare "zero" : !qlx.logical_qubit
    %m1 = qlx.measure <Z> %q2 : !qlx.logical_qubit -> i1
    qlx.discard %q1 : !qlx.logical_qubit
    qlx.return %m0, %m1 : i1, i1
  }
}

// CHECK: lvm.kernel @placed
// CHECK: lvm.prepare "zero" at @vm::@compute
// CHECK: lvm.measure
// WRITE-FAILURE: failed to write detached placement witness '/dev/full'

// WITNESS: {"schema":"qlx.placement-witness/v1"
// WITNESS-SAME: "root":"sequential"
// WITNESS-SAME: "domain":"vm"
// WITNESS-SAME: "objective":"first_fit"
// WITNESS-SAME: "bindings":[{"source":"prepare:0","space":"compute","slot":0},{"source":"prepare:3","space":"compute","slot":0}]}
// COMMIT: lvm.kernel @with_witness
// COMMIT-SAME: placement_witness_sha256 = "sha256:
// COMMIT-SAME: qlx.placement_policy = "native-first-fit/v3"
// CHECK: lvm.prepare "zero" at @vm::@compute
// CHECK: lvm.measure
// REUSE: "bindings":[{"source":"prepare:0","space":"compute","slot":0},{"source":"prepare:1","space":"compute","slot":1},{"source":"prepare:3","space":"compute","slot":0}]
