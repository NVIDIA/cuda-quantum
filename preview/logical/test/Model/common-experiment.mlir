// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p0"]} {
  qlx.program @memory : () -> () attributes {qlx.profile = "p0"} {
    qlx.return
  }
  qlx.experiment @memory_point {
    bindings = {parameters = {rounds = 3 : i64}},
    closure = [@memory],
    pass_recipe = [
      {name = "qlx-normalize-actions", options = {}},
      {name = "qlx-verify-p0", options = {strict = true}}
    ],
    stage = "p0", facets = [],
    root = @memory
  }
}

// CHECK: qlx.experiment @memory_point
// CHECK-SAME: closure = [@memory]
// CHECK-SAME: root = @memory
// CHECK-SAME: stage = "p0"
