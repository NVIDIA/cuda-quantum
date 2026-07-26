/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --quake-add-metadata | FileCheck %s

#include <cudaq.h>

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel
// CHECK-SAME: () attributes {
// CHECK-SAME: {{("cudaq-entrypoint".*qubitMeasurementFeedback = true.*[}])|([{].*qubitMeasurementFeedback = true.*"cudaq-entrypoint")}} {
// clang-format on
struct kernel {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    cnot(q[0], q[1]);
    h(q[0]);

    auto b0 = mz(q[0]);
    auto b1 = mz(q[1]);

    if (b1)
      x(q[2]);
    if (b0)
      z(q[2]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernelNoConditional
// CHECK-SAME: () attributes {{{.*}}"cudaq-entrypoint"{{.*}}} {
struct kernelNoConditional {
  void operator()() __qpu__ {
    cudaq::qarray<1> q;
    h(q[0]);
  }
};

// A measurement result consumed as a runtime `observable_index` is
// measurement feedback even though there is no control flow.
// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernelObservableIndexFeedback
// CHECK-SAME: () attributes {
// CHECK-SAME: {{("cudaq-entrypoint".*qubitMeasurementFeedback = true.*[}])|([{].*qubitMeasurementFeedback = true.*"cudaq-entrypoint")}} {
// clang-format on
struct kernelObservableIndexFeedback {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    auto ms = mz(q);
    cudaq::logical_observable(ms, static_cast<bool>(ms[0]));
  }
};

// A runtime `observable_index` that does not derive from a measurement (a
// loop induction variable) is NOT measurement feedback.
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernelObservableLoopIndex
// CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
struct kernelObservableLoopIndex {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    auto ms = mz(q);
    for (std::size_t i = 0; i < 2; i++)
      cudaq::logical_observable(ms, i);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernelComplex
// CHECK-SAME: () attributes {
// CHECK-SAME: {{("cudaq-entrypoint".*qubitMeasurementFeedback = true.*[}])|([{].*qubitMeasurementFeedback = true.*"cudaq-entrypoint")}} {
// clang-format on
struct kernelComplex {
  void operator()() __qpu__ {
    // Allocate the qubits
    cudaq::qvector q(2), ancilla(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);

    for (std::size_t i = 0; i < 2; i++) {

      // for each qubit, get a random result that
      // dictates the measurement basis.
      h(ancilla);

      bool mzA0 = mz(ancilla[0]);
      bool mzA1 = mz(ancilla[1]);

      if (mzA0 && mzA1) {
        h(q[i]);
      } else if (mzA0 && !mzA1) {
        s<cudaq::adj>(q[i]);
        h(q[i]);
      }

      reset(ancilla[0]);
      reset(ancilla[1]);
    }

    mz(q);
  }
};
