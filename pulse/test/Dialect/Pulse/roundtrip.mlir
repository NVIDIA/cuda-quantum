// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_qudit_and_drive
func.func @test_qudit_and_drive() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.drive
  %d1, %t1 = pulse.drive %d0, %wf, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// CHECK-LABEL: func @test_readout
func.func @test_readout() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %duration = arith.constant 1000 : i64
  %amplitude = arith.constant 0.05 : f64
  %zero = arith.constant 0.0 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.readout
  %r1, %rt1, %m = pulse.readout %r0, %wf, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}
