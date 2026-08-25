/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t
// clang-format on

#include <cassert>
#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>
#include <stdexcept>
#include <string>

#include <iostream>

struct kernelWithMeasure {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  }
};

int main() {

  bool threw = false;
  try {
    cudaq::estimate_resources(
        []() -> bool { throw std::runtime_error("choice failed"); },
        kernelWithMeasure{});
  } catch (const std::runtime_error &re) {
    assert(std::string(re.what()).find("choice failed") != std::string::npos);
    threw = true;
  } catch (...) {
    std::cout << "got an unrecognised error" << std::endl;
    threw = true;
  }
  assert(threw);

  // A subsequent estimate_resources on the same thread must work.
  auto resources = cudaq::estimate_resources(kernelWithMeasure{});
  assert(resources.count("h") == 1);

  return 0;
}
