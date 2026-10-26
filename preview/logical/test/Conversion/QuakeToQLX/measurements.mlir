// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --convert-quake-to-qlx --split-input-file %s | FileCheck %s --implicit-check-not=quake.

module {
  func.func @__nvqpp__mlirgen__measurement_axes() -> (i1, i1, i1)
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %1 = quake.null_wire
    %2 = quake.null_wire
    %mx, %wx = quake.mx %0 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %my, %wy = quake.my %1 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %mz, %wz = quake.mz %2 : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
    %x = quake.discriminate %mx : (!cc.measure_handle) -> i1
    %y = quake.discriminate %my : (!cc.measure_handle) -> i1
    %z = quake.discriminate %mz : (!cc.measure_handle) -> i1
    quake.sink %wx : !quake.wire
    quake.sink %wy : !quake.wire
    quake.sink %wz : !quake.wire
    return %x, %y, %z : i1, i1, i1
  }
}

// CHECK-LABEL: qlx.program @measurement_axes : () -> (i1, i1, i1)
// CHECK: %[[MEAS_QX:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[MEAS_QY:.*]] = qlx.prepare "zero" {allocation = 1 : i64
// CHECK: %[[MEAS_QZ:.*]] = qlx.prepare "zero" {allocation = 2 : i64
// CHECK: %[[MEAS_X:.*]] = qlx.measure <X> %[[MEAS_QX]]
// CHECK: %[[MEAS_Y:.*]] = qlx.measure <Y> %[[MEAS_QY]]
// CHECK: %[[MEAS_Z:.*]] = qlx.measure <Z> %[[MEAS_QZ]]
// CHECK: qlx.return %[[MEAS_X]], %[[MEAS_Y]], %[[MEAS_Z]] : i1, i1, i1

// -----

// CUDA-Q also accepts the legacy !quake.measure result form. Keep this case
// explicit: both upstream measurement representations have the same P0
// meaning and must cross the ingestion boundary.
module {
  func.func @__nvqpp__mlirgen__legacy_measurement() -> i1
      attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    %0 = quake.null_wire
    %m, %w = quake.mz %0 : (!quake.wire) -> (!quake.measure, !quake.wire)
    %bit = quake.discriminate %m : (!quake.measure) -> i1
    quake.sink %w : !quake.wire
    return %bit : i1
  }
}

// CHECK-LABEL: qlx.program @legacy_measurement : () -> i1
// CHECK: %[[LEGACY_Q:.*]] = qlx.prepare "zero" {allocation = 0 : i64
// CHECK: %[[LEGACY_M:.*]] = qlx.measure <Z> %[[LEGACY_Q]]
// CHECK: qlx.return %[[LEGACY_M]] : i1
