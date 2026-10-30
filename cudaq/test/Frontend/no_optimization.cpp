/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --add-dealloc | \
// RUN:   cudaq-translate --convert-to=qir-base | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --inline --device-code-loader | \
// RUN:   FileCheck --check-prefix=LOADER %s
// RUN: cudaq-quake --quantum-opt-level=2 %s | cudaq-opt --inline \
// RUN:   --device-code-loader | FileCheck --check-prefix=LEVEL2 %s

#include "cudaq/qis/qubit_qis.h"

struct Defeatism {
  void operator()() __disable_quantum_optimization__ __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);
    h(q0);
  }
};

__qpu__ __disable_quantum_optimization__ void echo(cudaq::qubit &q) {
  h(q);
  h(q);
}

// nvq++ inlines `echo` before it registers `Caller` for the JIT, so `Caller`
// must carry the marker itself to keep the gates of `echo`.
struct Caller {
  void operator()() __qpu__ {
    cudaq::qubit q;
    echo(q);
  }
};

// A kernel that calls no marked kernel is registered without the marker, even
// though the source file's module has it.
struct Bystander {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    h(q);
  }
};

// clang-format off
// CHECK-LABEL: define void @__nvqpp__mlirgen__Defeatism()
// CHECK:         tail call void @__quantum__qis__h
// CHECK:         tail call void @__quantum__qis__h
// CHECK-NOT:     @__quantum__qis__h
// CHECK:         ret void

// LOADER:      @CallerCodeHolder.extract_device_code("module attributes
// LOADER-SAME:   quake.optimizationLevel = 0 : i32
// LOADER-SAME:   func.func @__nvqpp__mlirgen__Caller() attributes {{.*}}quake.optimizationLevel = 0 : i32
// LOADER:      @BystanderCodeHolder.extract_device_code("module attributes
// LOADER-NOT:    quake.optimizationLevel
// LOADER-SAME:   func.func @__nvqpp__mlirgen__Bystander()

// With `-fquantum-opt-level=2`, the translation unit's module carries level 2.
// A kernel that (transitively) calls a marked kernel is registered at level 0,
// while the other kernels keep the flag's level.
// LEVEL2:      @CallerCodeHolder.extract_device_code("module attributes
// LEVEL2-SAME:   quake.optimizationLevel = 0 : i32
// LEVEL2:      @BystanderCodeHolder.extract_device_code("module attributes
// LEVEL2-SAME:   quake.optimizationLevel = 2 : i32
// LEVEL2-SAME:   func.func @__nvqpp__mlirgen__Bystander()
