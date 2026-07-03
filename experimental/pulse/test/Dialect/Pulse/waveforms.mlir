// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_waveforms
func.func @test_waveforms() {
  // CHECK: pulse.square
  %sq = pulse.square 100, [0.5, 0.0] : !pulse.waveform
  // CHECK: pulse.gaussian
  %g = pulse.gaussian 40, 0.3, 10.0 : !pulse.waveform
  // CHECK: pulse.gaussian_square
  %gs = pulse.gaussian_square 200, 0.4, 8.0, 20 : !pulse.waveform
  // CHECK: pulse.drag
  %dr = pulse.drag 40, 0.25, 10.0, 0.5 : !pulse.waveform
  // CHECK: pulse.cosine
  %cos = pulse.cosine 100, 0.3 : !pulse.waveform
  // CHECK: pulse.tanh_ramp
  %ramp = pulse.tanh_ramp 80, 0.5, 5.0 : !pulse.waveform
  return
}
