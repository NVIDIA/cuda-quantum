// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_qudit_and_drive
func.func @test_qudit_and_drive() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %wf = pulse.gaussian 40, 0.3, 10.0 : !pulse.waveform
  // CHECK: pulse.drive
  %d1, %t1 = pulse.drive %d0, %wf, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// CHECK-LABEL: func @test_readout
func.func @test_readout() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %wf = pulse.square 1000, [0.05, 0.0] : !pulse.waveform
  // CHECK: pulse.readout
  %r1, %rt1, %m = pulse.readout %r0, %wf, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}
