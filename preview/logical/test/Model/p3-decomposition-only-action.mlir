// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

// Parameterized rotations and movement primitives still declare a process.
// A provider may fail closed on an unsupported builtin process name.

// CHECK: phys.action @rz
// CHECK-SAME: parameters = ["theta"]
// CHECK-SAME: process =
phys.action @rz {
  arity = 1 : i64,
  parameters = ["theta"],
  process = "{\22kind\22:\22builtin\22,\22name\22:\22rz\22,\22parameters\22:{}}"
}

// CHECK: phys.action @shuttle
// CHECK-SAME: process =
phys.action @shuttle {
  arity = 1 : i64,
  process = "{\22kind\22:\22builtin\22,\22name\22:\22transport\22,\22parameters\22:{}}"
}
