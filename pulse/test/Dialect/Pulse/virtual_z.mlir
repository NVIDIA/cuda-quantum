// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-virtual-z %s | FileCheck %s

// CHECK-LABEL: func.func @fold_shift_into_drive
func.func @fold_shift_into_drive() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 0.785398163397448 : f64
  // The shift_phase should be folded into the drive as a persistent frame attr
  %t2 = pulse.shift_phase %t, %ph : !pulse.tone, f64 -> !pulse.tone
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 3.000000e-01 : f64
  %sigma = arith.constant 1.000000e+01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  // CHECK: = pulse.drive
  // CHECK-SAME: frame_phase_offset
  %d2, %t3 = pulse.drive %d, %wf, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK-NOT: pulse.shift_phase
  return
}
