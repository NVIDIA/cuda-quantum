/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simple test to make sure the tool is built and has basic functionality.

// RUN: cudaq-quake --emit-llvm-file %s | FileCheck %s

// We should only have 2 kernels created even though we instantiate ghz<5> twice. 
// CHECK-LABEL: module attributes {quake.mangled_name_map = {
// CHECK-SAME: __nvqpp__mlirgen__ghzILm10EE = "_ZN3ghzILm10EEclEv", __nvqpp__mlirgen__ghzILm5EE = "_ZN3ghzILm5EEclEv"

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ghzILm5EE
// CHECK: quake.h
// CHECK: cc.loop while {
// CHECK: quake.x
// CHECK: quake.mz

// CHECK-LABEL: func.func @__nvqpp__mlirgen__ghzILm10EE
// CHECK: quake.h
// CHECK: cc.loop while {
// CHECK: quake.x
// CHECK: quake.mz


#include <cudaq.h>
#include <cudaq/algorithm.h>

// Define a quantum kernel
template <std::size_t N> struct ghz {
  auto operator()() __qpu__ {
    cudaq::qreg<N> q;
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  // Run the kernel in NISQ mode (i.e. run and
  // collect bit strings and counts)
  ghz<5> g;
  auto counts = cudaq::sample(g);
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  ghz<5> gg;
  auto counts2 = cudaq::sample(gg);
  counts2.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts2) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }

  ghz<10> ggg;
  auto counts3 = cudaq::sample(ggg);
  counts3.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts3) {
    printf("Observed: %s, %lu\n", bits.c_str(), count);
  }
  return 0;
}
