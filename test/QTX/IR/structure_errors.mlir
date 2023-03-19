// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{op has 1 target operand(s), but returns 0 target(s)}}
qtx.circuit @return_wire(%wire : !qtx.wire) {
  %new_wire = h %wire : !qtx.wire
  return
}

// -----

// expected-error@+1 {{op has mismatching input(s) and result(s) target(s)}}
qtx.circuit @return_wire(%wire : !qtx.wire) -> (!qtx.wire_array<1>) {
  %new_wire = h %wire : !qtx.wire
  return
}

// -----

qtx.circuit @return_wire(%wire : !qtx.wire) -> (!qtx.wire) {
  %array = alloca : !qtx.wire_array<1>
  %new_wire = h %wire : !qtx.wire
 // expected-error@+1 {{type of return target operand 0 ('!qtx.wire_array<1>') doesn't match circuit target result type ('!qtx.wire') in circuit @return_wire}}
  return %array : !qtx.wire_array<1>
}

// -----

// expected-error@+1 {{failed parsing the result types}}
qtx.circuit @rz<%a: f32>(%w0: !qtx.wire) -> !qtx.wire {
  %w1 = rz<%a> %w0 : <f32> !qtx.wire
  return %w1 : !qtx.wire
}
