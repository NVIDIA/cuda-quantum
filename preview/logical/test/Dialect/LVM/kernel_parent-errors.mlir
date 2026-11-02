// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s -split-input-file -verify-diagnostics

lvm.domain @vm {
  lvm.space @compute {capabilities = []}
}
func.func @bad_apply(%q: !lvm.logical_qubit<@vm::@compute>)
    -> !lvm.logical_qubit<@vm::@compute> {
  // expected-error @+1 {{must appear inside lvm.kernel}}
  %out = lvm.apply #qlx.action<h>(%q) at [@vm::@compute]
      : (!lvm.logical_qubit<@vm::@compute>)
        -> !lvm.logical_qubit<@vm::@compute>
  func.return %out : !lvm.logical_qubit<@vm::@compute>
}

// -----

lvm.domain @vm {
  lvm.space @memory {capabilities = []}
}
func.func @bad_idle(%q: !lvm.logical_qubit<@vm::@memory>)
    -> !lvm.logical_qubit<@vm::@memory> {
  %rounds = arith.constant 1 : index
  // expected-error @+1 {{must appear inside lvm.kernel}}
  %out = lvm.idle %q rounds %rounds at [@vm::@memory]
      : (!lvm.logical_qubit<@vm::@memory>)
        -> !lvm.logical_qubit<@vm::@memory>
  func.return %out : !lvm.logical_qubit<@vm::@memory>
}
