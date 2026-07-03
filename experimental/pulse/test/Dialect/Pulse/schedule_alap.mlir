// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s

// CHECK-LABEL: func.func @simple_schedule
func.func @simple_schedule() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d, %t = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %wf = pulse.gaussian 40, 3.000000e-01, 1.000000e+01 : !pulse.waveform
  // CHECK: pulse.drive
  // CHECK-SAME: start_vtu = 0
  // CHECK-SAME: duration_vtu = 40
  %d2, %t2 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
