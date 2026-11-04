// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --allow-unregistered-dialect --qlx-estimate-logical 2>&1 | FileCheck %s
// RUN: not qlx-opt %s --allow-unregistered-dialect --qlx-estimate-logical='root=third_party' 2>&1 | FileCheck %s --check-prefix=THIRD-PARTY

qlx.program @first : () -> () attributes {qlx.stage = "p0"} {
  qlx.return
}

qlx.program @second : () -> () attributes {qlx.stage = "p0"} {
  qlx.return
}

qlx.program @third_party : (!qlx.logical_qubit) -> !qlx.logical_qubit attributes {qlx.stage = "p0"} {
^bb0(%q: !qlx.logical_qubit):
  %out = "vendor.magic"(%q) : (!qlx.logical_qubit) -> !qlx.logical_qubit
  qlx.return %out : !qlx.logical_qubit
}

// CHECK: error: qlx-estimate-logical requires root= when several qlx.program symbols exist
// THIRD-PARTY: has no typed cost semantics for this operation on logical quantum values
