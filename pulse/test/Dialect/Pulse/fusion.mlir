// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-fusion %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_adjacent_squares
func.func @fuse_adjacent_squares() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  // Two adjacent square pulses with same amplitude should fuse
  %duration = arith.constant 50 : i64
  %amplitude = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %wf1 = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf1, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %wf2 = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d3, %t3 = pulse.drive %d2, %wf2, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: arith.constant 100
  // CHECK: pulse.square
  // CHECK: fused
  return
}
