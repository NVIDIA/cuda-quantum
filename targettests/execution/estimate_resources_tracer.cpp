/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// FIXME: This is several distinct tests. Split it into multiple files as it is
// slow and serializing it like this makes it a long pole test.

// RUN: nvq++ -DCASE1 %s -o %t && CUDAQ_LOG_LEVEL=info %t | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// RUN: nvq++ -DCASE2 %s -o %t && CUDAQ_LOG_LEVEL=info %t | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// RUN: nvq++ -DCASE3 %s -o %t && CUDAQ_LOG_LEVEL=info %t | \
// RUN:   FileCheck %s --check-prefixes=CHECK,AOT
// RUN: nvq++ --target quantinuum --emulate -DCASE1 %s -o %t && \
// RUN:   CUDAQ_LOG_LEVEL=info %t | FileCheck %s --check-prefixes=CHECK,EMULATE
// RUN: nvq++ --target quantinuum --emulate -DCASE2 %s -o %t && \
// RUN:   CUDAQ_LOG_LEVEL=info %t | FileCheck %s --check-prefixes=CHECK,EMULATE

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

// EMULATE: JIT high level:
// EMULATE: Pass pipeline for
// EMULATE-NOT: Applying x with 1 controls

#include <cassert>
#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>
#include <numeric>

#ifdef CASE1

int main() {
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

#elif CASE2

int main() {
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

#elif CASE3

#include <random>
#include <vector>

int main() {
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
