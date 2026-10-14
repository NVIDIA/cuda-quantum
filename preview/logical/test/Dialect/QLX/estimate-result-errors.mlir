// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s 2>&1 | FileCheck %s

qlx.program @work : () -> () attributes {qlx.stage = "p0"} {
  qlx.return
}

qlx.estimate_result @bad {
  assumptions = [],
  data = {},
  evidence = [],
  root = @work,
  schema = "qlx.logical-profile/v1",
  tier = "future"
}

// CHECK: error: 'qlx.estimate_result' op tier must be logical, static, analytical, schedule, or twin
