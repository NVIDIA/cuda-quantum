// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// CHECK-LABEL: func @test_timing
func.func @test_timing() {
  %q0 = pulse.qudit_alloc : !pulse.qref
  %q1 = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)

  %c20 = arith.constant 20 : i64
  %dur = pulse.duration_from_int %c20 : (i64) -> !pulse.duration
  // CHECK: pulse.wait
  %d0a = pulse.wait %d0, %dur : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line

  // CHECK: pulse.sync
  %d0b, %d1a = pulse.sync %d0a, %d1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
  return
}
