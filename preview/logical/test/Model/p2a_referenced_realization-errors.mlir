// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.circuit @body(%value: i1) -> i1 {
    fabric.return %value : i1
  }
  // expected-error @+1 {{realization circuit signature must match the gadget signature}}
  fabric.gadget @bad(%value: i64) -> i64 realization @body
}

// -----

module {
  fabric.circuit @body(%value: i1) -> i1 {
    fabric.return %value : i1
  }
  // expected-error @+1 {{has both an inline body and a realization circuit reference}}
  "fabric.gadget"() <{function_type = (i1) -> i1, realization = @body,
                       sym_name = "bad"}> ({
  ^bb0(%value: i1):
    fabric.return %value : i1
  }) : () -> ()
}

// -----

module {
  // expected-error @+2 {{operand types must match enclosing circuit result types}}
  fabric.circuit @bad(%value: i1) -> i64 {
    fabric.return %value : i1
  }
}
