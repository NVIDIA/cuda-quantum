// RUN: cudaq-pulse-opt --pulse-fusion %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_adjacent_squares
func.func @fuse_adjacent_squares() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  // Two adjacent square pulses with same amplitude should fuse
  %wf1 = pulse.square 50, [2.000000e-01, 0.000000e+00] : !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf1, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %wf2 = pulse.square 50, [2.000000e-01, 0.000000e+00] : !pulse.waveform
  %d3, %t3 = pulse.drive %d2, %wf2, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: pulse.square 100
  // CHECK: fused
  return
}
