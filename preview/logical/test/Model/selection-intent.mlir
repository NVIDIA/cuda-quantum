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
  event.selection %accept {mode = "require"} : i1
  event.selection %accept {mode = "condition_results"} : i1
  event.selection %accept {mode = "abort_on", accept_when = false} : i1
}

// CHECK: event.selection {{.*}} {mode = "require"} : i1
// CHECK: event.selection {{.*}} {mode = "condition_results"} : i1
// CHECK: event.selection {{.*}} {accept_when = false, mode = "abort_on"} : i1
