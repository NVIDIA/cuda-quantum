// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_waveforms
func.func @test_waveforms() {
  %c0 = arith.constant 0.0 : f64
  %c025 = arith.constant 0.25 : f64
  %c03 = arith.constant 0.3 : f64
  %c04 = arith.constant 0.4 : f64
  %c05 = arith.constant 0.5 : f64
  %c5 = arith.constant 5.0 : f64
  %c8 = arith.constant 8.0 : f64
  %c10 = arith.constant 10.0 : f64
  %c20 = arith.constant 20 : i64
  %c40 = arith.constant 40 : i64
  %c80 = arith.constant 80 : i64
  %c100 = arith.constant 100 : i64
  %c200 = arith.constant 200 : i64
  // CHECK: pulse.square
  %sq = pulse.square %c100, %c05, %c0 : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.gaussian
  %g = pulse.gaussian %c40, %c03, %c10 : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.gaussian_square
  %gs = pulse.gaussian_square %c200, %c04, %c8, %c20
      : i64, f64, f64, i64 -> !pulse.waveform
  // CHECK: pulse.drag
  %dr = pulse.drag %c40, %c025, %c10, %c05
      : i64, f64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.cosine
  %cos = pulse.cosine %c100, %c03 : i64, f64 -> !pulse.waveform
  // CHECK: pulse.tanh_ramp
  %ramp = pulse.tanh_ramp %c80, %c05, %c5
      : i64, f64, f64 -> !pulse.waveform
  return
}
