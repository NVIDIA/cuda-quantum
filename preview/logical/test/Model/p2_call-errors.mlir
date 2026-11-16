// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.protocol @callee : (i1) -> i1 attributes {objective = @identity} {
  ^bb0(%value: i1):
    fabric.protocol_return %value : i1
  }
  fabric.protocol @caller : (i64) -> i64 attributes {objective = @identity} {
  ^bb0(%value: i64):
    // expected-error @+1 {{operand/result types must match the callee signature}}
    %result = fabric.call @callee(%value) : (i64) -> i64
    fabric.protocol_return %result : i64
  }
}
