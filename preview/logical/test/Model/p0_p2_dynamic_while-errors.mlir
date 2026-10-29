// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  func.func @bad_bound(%go: i1) {
    // expected-error @+1 {{attribute 'max_iterations' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
    %0 = "cflow.while"(%go) <{max_iterations = 0 : i64}> ({
    ^bb0(%current: i1):
      "cflow.while_condition"(%current, %current) : (i1, i1) -> ()
    }, {
    ^bb0(%current: i1):
      cflow.yield %current : i1
    }) : (i1) -> i1
    return
  }
}

// -----

module {
  func.func @bad_condition_terminator(%go: i1) {
    // expected-error @+1 {{before region must terminate with cflow.while_condition}}
    %0 = "cflow.while"(%go) ({
    ^bb0(%current: i1):
      cflow.yield %current : i1
    }, {
    ^bb0(%current: i1):
      cflow.yield %current : i1
    }) : (i1) -> i1
    return
  }
}

// -----

module {
  func.func @bad_forwarded_type(%go: i1) {
    // expected-error @+1 {{while_condition forwarded types must match loop result types}}
    %0 = "cflow.while"(%go) ({
    ^bb0(%current: i1):
      %wrong = arith.constant 0 : i64
      "cflow.while_condition"(%current, %wrong) : (i1, i64) -> ()
    }, {
    ^bb0(%current: i1):
      cflow.yield %current : i1
    }) : (i1) -> i1
    return
  }
}
