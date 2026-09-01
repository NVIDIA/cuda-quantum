// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module {
  %accept = arith.constant true
  qlx.selection %accept {mode = "require"} : i1
  lvm.selection %accept {mode = "condition_results"} : i1
  fabric.selection %accept {mode = "abort_on", accept_when = false} : i1
}

// CHECK: qlx.selection {{.*}} {mode = "require"} : i1
// CHECK: lvm.selection {{.*}} {mode = "condition_results"} : i1
// CHECK: fabric.selection {{.*}} {accept_when = false, mode = "abort_on"} : i1
