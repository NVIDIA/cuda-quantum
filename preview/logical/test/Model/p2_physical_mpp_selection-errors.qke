// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  func.func @bad_width(%patch: !fabric.patch<@code, @encoding, @epoch>) {
    // expected-error @+1 {{'fabric.mpp' op paulis width must equal indices width}}
    %next, %bit = fabric.mpp %patch data indices [0, 2] paulis "X"
      : !fabric.patch<@code, @encoding, @epoch> -> tensor<1xi1>
    return
  }
}

// -----

module {
  func.func @bad_pauli(%patch: !fabric.patch<@code, @encoding, @epoch>) {
    // expected-error @+1 {{'fabric.mpp' op paulis must contain only X, Y, and Z}}
    %next, %bit = fabric.mpp %patch data indices [0] paulis "I"
      : !fabric.patch<@code, @encoding, @epoch> -> tensor<1xi1>
    return
  }
}

// -----

module {
  func.func @empty_parity() {
    // expected-error @+1 {{'fabric.parity' op requires one or more measurement bundles}}
    %event = fabric.parity : () -> i1
    return
  }
}

// -----

module {
  func.func @empty_events() {
    // expected-error @+1 {{'fabric.all_false' op requires one or more classical events}}
    %accepted = fabric.all_false : () -> i1
    return
  }
}
