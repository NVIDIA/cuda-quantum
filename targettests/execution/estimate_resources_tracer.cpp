/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile once per target; run each case in its own process.

// RUN: nvq++ %s -o %t
// RUN: CUDAQ_LOG_LEVEL=info %t 1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// RUN: CUDAQ_LOG_LEVEL=info %t 2 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// RUN: CUDAQ_LOG_LEVEL=info %t 3 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// RUN: CUDAQ_LOG_LEVEL=info %t 4 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// clang-format off
// RUN: if %quantinuum_avail; then nvq++ --target quantinuum --emulate -DEMULATE %s -o %t.emulate; fi
// RUN: if %quantinuum_avail; then CUDAQ_LOG_LEVEL=info %t.emulate 1 | FileCheck %s --check-prefixes=CHECK,EMULATE; fi
// RUN: if %quantinuum_avail; then CUDAQ_LOG_LEVEL=info %t.emulate 2 | FileCheck %s --check-prefixes=CHECK,EMULATE; fi
// RUN: if %quantinuum_avail; then CUDAQ_LOG_LEVEL=info %t.emulate 4 | \
// RUN:   FileCheck %s --check-prefixes=CHECK,EMULATE-ATOMIC; fi
// clang-format on

// We don't run CASE3 with emulation on because compilation takes several
// minutes

// CHECK: Launching kernel with estimate policy
// CHECK: Launching kernel in sync mode with policy resource-count
// CHECK: No compiled module found. Compiling.

// When using AOT compilation, we expect no JIT compilation, with all gate
// tracing happening at runtime

// AOT: No JIT compilation required. Using AOT-compiled module as-is.
// AOT: Applying x with 1 controls

// When using JIT compilation, we expect JIT compilation, but all tracing gets
// folded away at JIT compile time

// EMULATE: Pass pipeline for
// EMULATE-NOT: Applying x with 1 controls

// EMULATE-ATOMIC: Pass pipeline for
// EMULATE-ATOMIC: Applying h with 0 controls
// EMULATE-ATOMIC: Applying x with 1 controls

#include "cudaq/qis/qubit_qis.h"
#include <cassert>
#include <cstdlib>
#include <cudaq/algorithms/resource_estimation.h>
#include <numeric>

int case1() {
  auto bellKernel = []() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  };

  auto resources = cudaq::estimate_resources(bellKernel);
  assert(resources.count("h") == 1);
  assert(resources.count_controls("x", /*controls*/ 1) == 1);
  assert(resources.count("rx") == 0);
  return 0;
}

int case2() {
  auto ghzKernel = [](int i) __qpu__ {
    cudaq::qvector q(i);
    h(q[0]);
    for (int j = 0; j < i - 1; j++)
      x<cudaq::ctrl>(q[j], q[j + 1]);
  };

  auto resources = cudaq::estimate_resources(ghzKernel, 10);
  assert(resources.count("h") == 1);
  assert(resources.count_controls("x", /*nControls*/ 1) == 9);
  assert(resources.count("x") == 9);
  return 0;
}

#ifndef EMULATE

#include <random>
#include <vector>

int case3() {
  auto largeTraceKernel = [](int numQubits, int numLayers,
                             std::vector<int> &cnotPairs) __qpu__ {
    cudaq::qvector q(numQubits);

    for (int layer = 0; layer < numLayers; layer++) {
      for (int i = 0; i < numQubits; i++) {
        int choice = cnotPairs[layer * numQubits + i] % 6;
        if (choice == 0)
          h(q[i]);
        else if (choice == 1)
          x(q[i]);
        else if (choice == 2)
          y(q[i]);
        else if (choice == 3)
          z(q[i]);
        else if (choice == 4)
          s(q[i]);
        else
          t(q[i]);
      }

      for (int i = 0; i < numQubits; i += 2)
        x<cudaq::ctrl>(q[i], q[i + 1]);
    }
  };

  int numQubits = 1000;
  int numLayers = 1000;

  std::vector<int> cnots(numQubits * numLayers);
  std::iota(cnots.begin(), cnots.end(), 0);
  std::shuffle(cnots.begin(), cnots.end(),
               std::mt19937{std::random_device{}()});
  auto resources =
      cudaq::estimate_resources(largeTraceKernel, numQubits, numLayers, cnots);
  auto totalOps = resources.count();
  assert(totalOps == numLayers * numQubits * 1.5);

  return 0;
}

#endif

struct atomic_workload {
  void operator()(cudaq::qview<> q) __qpu__ __atomic_quantum_region__ {
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
  }
};

struct atomic_round_trip {
  void operator()() __qpu__ {
    cudaq::qvector q(3);
    atomic_workload{}(q);
    cudaq::adjoint(atomic_workload{}, q);
    mz(q);
  }
};

int case4() {
  auto resources = cudaq::estimate_resources(atomic_round_trip{});
  assert(resources.count("h") == 2);
  assert(resources.count_controls("x", /*controls*/ 1) == 4);
  assert(resources.count("mz") == 3);
  assert(resources.getNumQubits() == 3);
  assert(resources.getCircuitDepth() == 7);
  assert(resources.getMultiQubitDepth() == 4);
  return 0;
}

int main(int argc, char **argv) {
  assert(argc == 2);
  switch (std::atoi(argv[1])) {
  case 1:
    return case1();
  case 2:
    return case2();
#ifndef EMULATE
  case 3:
    return case3();
#endif
  case 4:
    return case4();
  default:
    return 1;
  }
}
