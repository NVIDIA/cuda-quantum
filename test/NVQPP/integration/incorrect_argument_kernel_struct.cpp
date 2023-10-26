/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s 2>&1 | FileCheck %s

#include <cudaq.h>

template <std::size_t N>
struct ghz {
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
  // Provide an argument but not needed.
  auto result = cudaq::sample(ghz<3>{}, 1.234);
}

// CHECK: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <>, got <double>"}>'
