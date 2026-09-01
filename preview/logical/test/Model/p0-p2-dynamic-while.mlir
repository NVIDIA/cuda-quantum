// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s
// RUN: not qlx-opt %s --fabric-count='root=qec' 2>&1 | FileCheck %s --check-prefix=COUNT-ERR

module attributes {qlx.profiles = ["p0", "p1", "p2n"]} {
  qlx.program @portable : (i1) -> i1 attributes {qlx.profile = "p0"} {
  ^bb0(%go: i1):
    %0 = "qlx.while"(%go) <{max_iterations = 8 : i64}> ({
    ^bb0(%current: i1):
      "qlx.while_condition"(%current, %current) : (i1, i1) -> ()
    }, {
    ^bb0(%current: i1):
      qlx.yield %current : i1
    }) : (i1) -> i1
    qlx.return %0 : i1
  }

  lvm.domain @machine {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  lvm.kernel @placed on @machine : (i1) -> i1 attributes {qlx.profile = "p1"} {
  ^bb0(%go: i1):
    %0 = "lvm.while"(%go) <{max_iterations = 8 : i64}> ({
    ^bb0(%current: i1):
      "lvm.while_condition"(%current, %current) : (i1, i1) -> ()
    }, {
    ^bb0(%current: i1):
      lvm.yield %current : i1
    }) : (i1) -> i1
    lvm.return %0 : i1
  }

  fabric.protocol @qec : (i1) -> i1 {
  ^bb0(%go: i1):
    %0 = "fabric.while"(%go) <{max_iterations = 8 : i64}> ({
    ^bb0(%current: i1):
      "fabric.while_condition"(%current, %current) : (i1, i1) -> ()
    }, {
    ^bb0(%current: i1):
      fabric.yield %current : i1
    }) : (i1) -> i1
    fabric.protocol_return %0 : i1
  }
}

// CHECK: "qlx.while"
// CHECK-SAME: max_iterations = 8 : i64
// CHECK: "qlx.while_condition"
// CHECK: "lvm.while"
// CHECK: "lvm.while_condition"
// CHECK: "fabric.while"
// CHECK: "fabric.while_condition"
// COUNT-ERR: fabric-count does not support this dynamic or unrecognized region-bearing executable operation
