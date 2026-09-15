// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-to-qop --qop-to-cudm --cudm-to-llvm \
// RUN:   --canonicalize --convert-arith-to-llvm --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts %s | FileCheck %s --check-prefix=FULL

// FULL-LABEL: llvm.func @main
// FULL: llvm.call @cudm_init
// FULL: llvm.call @cudm_evolve
// FULL: llvm.call @cudm_state_capture
// FULL-NOT: pulse.drive
// FULL-NOT: qop.lindblad
// FULL-NOT: cudm.evolve

// CHECK-LABEL: func @main
func.func @main() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.3 : f64
  %sigma = arith.constant 10.0 : f64
  %wf = pulse.gaussian %duration, %amplitude, %sigma
      : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %wf, %t0
      {start_vtu = 0 : i64, duration_vtu = 40 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone

  // CHECK: qop.spin
  // CHECK: qop.make_product
  // CHECK: qop.callback_scalar
  // CHECK: qop.make_sum
  // CHECK: qop.lindblad
  return
}
