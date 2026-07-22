// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s

module @dissipator_test attributes {
    pulse.t1_times = [50.0e3 : f64],
    pulse.t2_times = [30.0e3 : f64]
} {

// CHECK-LABEL: func @main
func.func @main() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.0e9 : f64}
      : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %wf = pulse.gaussian 40, 0.3, 10.0 : !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %wf, %t0 {duration_vtu = 40 : i64}
      : !pulse.drive_line, !pulse.waveform, !pulse.tone
      -> !pulse.drive_line, !pulse.tone

  // CHECK: qop.spin{{.*}}spin_lowering
  // CHECK: qop.spin{{.*}}spin_z
  // CHECK: qop.lindblad
  return
}

}
