// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --qlx-estimate-logical 2>&1 | FileCheck %s

qlx.program @dynamic : (i1) -> i1 attributes {qlx.stage = "p0"} {
^bb0(%condition: i1):
  %0 = "qlx.while"(%condition) <{max_iterations = 8 : i64}> ({
  ^bb0(%current: i1):
    "qlx.while_condition"(%current, %current) : (i1, i1) -> ()
  }, {
  ^bb0(%current: i1):
    qlx.yield %current : i1
  }) : (i1) -> i1
  qlx.return %0 : i1
}

// CHECK: qlx-estimate-logical requires dynamic while control to be specialized to an exact folded form
