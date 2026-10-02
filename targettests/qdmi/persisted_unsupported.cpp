/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: qdmi
// clang-format off
// RUN: nvq++ --target qdmi --qdmi-device mqt.ddsim.default %s -o %t
// RUN: %t submit %t.future
// RUN: not %t reopen %t.future 2>&1 | FileCheck %s
// clang-format on

#include <cudaq.h>

#include <fstream>
#include <stdexcept>
#include <string_view>

struct simple_x {
  void operator()() __qpu__ {
    cudaq::qubit qubit;
    x(qubit);
    mz(qubit);
  }
};

int main(const int argc, char **argv) {
  if (argc != 3)
    throw std::invalid_argument("expected submit/reopen and a future path");

  if (std::string_view(argv[1]) == "submit") {
    auto future = cudaq::sample_async(16, 0, simple_x{});
    std::ofstream output(argv[2]);
    output << future;
    return 0;
  }

  cudaq::async_sample_result future;
  std::ifstream input(argv[2]);
  input >> future;
  static_cast<void>(future.get());
}

// CHECK: Retrieving job: Not supported.
