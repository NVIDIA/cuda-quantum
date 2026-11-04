/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// The `.run` entry point that `cudaq::run` launches carries the optimization
// level declared on the kernel it wraps. The level of the compilation unit is
// passed to the runtime by the launch instead.

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --kernel-execution=generate-run-stack=1 | \
// RUN:   FileCheck %s
// RUN: cudaq-quake %s | \
// RUN:   cudaq-opt --kernel-execution="generate-run-stack=1 quantum-opt-level=2" | \
// RUN:   FileCheck --check-prefix=LEVEL2 %s
// clang-format on

#include "cudaq/qis/qubit_qis.h"

__qpu__ __disable_quantum_optimization__ bool marked() {
  cudaq::qubit q;
  h(q);
  h(q);
  return mz(q);
}

__qpu__ bool unmarked() {
  cudaq::qubit q;
  h(q);
  h(q);
  return mz(q);
}

// clang-format off
// CHECK:  func.func @__nvqpp__mlirgen__function_marked._Z6markedv.run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i1], quake.optimizationLevel = 0 : i32} {
// CHECK:  func.func @__nvqpp__mlirgen__function_unmarked._Z8unmarkedv.run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i1]} {

// LEVEL2: func.func @__nvqpp__mlirgen__function_marked._Z6markedv.run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i1], quake.optimizationLevel = 0 : i32} {
// LEVEL2: func.func @__nvqpp__mlirgen__function_unmarked._Z8unmarkedv.run() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this, quake.cudaq_run = [i1]} {
// LEVEL2: func.func @__nvqpp__mlirgen__function_unmarked._Z8unmarkedv.run.entry(
// LEVEL2:   %[[LEVEL:.*]] = arith.constant 2 : i32
// LEVEL2:   call @hybridLaunchKernel({{.*}}, %[[LEVEL]]) : ({{.*}}, i32) ->
