// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_waveform_algebra
func.func @test_waveform_algebra() {
  %a = pulse.square 40, [0.3, 0.0] : !pulse.waveform
  %b = pulse.square 40, [0.1, 0.0] : !pulse.waveform

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
