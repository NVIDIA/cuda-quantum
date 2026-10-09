// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_waveform_algebra
func.func @test_waveform_algebra() {
  %duration = arith.constant 40 : i64
  %amp_a = arith.constant 0.3 : f64
  %amp_b = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %a = pulse.square %duration, %amp_a, %zero
      : i64, f64, f64 -> !pulse.waveform
  %b = pulse.square %duration, %amp_b, %zero
      : i64, f64, f64 -> !pulse.waveform

  // CHECK: pulse.add
  %sum = pulse.add %a, %b : !pulse.waveform
  // CHECK: pulse.sub
  %diff = pulse.sub %a, %b : !pulse.waveform
  // CHECK: pulse.mul
  %prod = pulse.mul %a, %b : !pulse.waveform

  %c = arith.constant 2.0 : f64
  // CHECK: pulse.scale
  %scaled = pulse.scale %a, %c : !pulse.waveform, f64 -> !pulse.waveform
  // CHECK: pulse.neg
  %negated = pulse.neg %a : !pulse.waveform
  return
}
