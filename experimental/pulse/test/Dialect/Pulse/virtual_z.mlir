// RUN: cudaq-pulse-opt --pulse-virtual-z %s | FileCheck %s

// CHECK-LABEL: func.func @fold_shift_into_drive
func.func @fold_shift_into_drive() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 0.785398163397448 : f64
  // The shift_phase should be folded into the drive as a phase_offset attr
  %t2 = pulse.shift_phase %t, %ph : !pulse.tone, f64 -> !pulse.tone
  %wf = pulse.gaussian 40, 3.000000e-01, 1.000000e+01 : !pulse.waveform
  // CHECK: pulse.drive
  // CHECK-SAME: phase_offset
  %d2, %t3 = pulse.drive %d, %wf, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK-NOT: pulse.shift_phase
  return
}
